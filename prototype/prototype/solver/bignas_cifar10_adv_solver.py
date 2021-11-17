import argparse
import time
import datetime
from PIL import Image
import torch
import math
import pprint
import torch.nn.functional as F
import copy
import random
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data.sampler import Sampler

from prototype.solver.base_solver import BaseSolver
from prototype.solver.cls_solver import ClsSolver
from prototype.model import model_entry
from prototype.data.sampler import DistributedSampler
from prototype.data.auto_augmentation import CIFAR10Policy, Cutout

from prototype.lr_scheduler import scheduler_entry
from prototype.utils.misc import makedir, AverageMeter, accuracy, load_state_model, mixup_data, \
    mix_criterion, cutmix_data
from prototype.utils.misc import accuracy, AverageMeter, mixup_data, cutmix_data, mix_criterion

try:
    from prototype.spring.nas.bignas.controller import ClsController
except ModuleNotFoundError:
    print('prototype.spring.nas not detected yet, install spring module first.')

torch.multiprocessing.set_sharing_strategy('file_system')

import pdb

# Adversarial Training
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class PGD():
    def __init__(self, eps, steps=7, alpha=None, loss_fn=None, targeted=False):
        '''
        Args:
            eps: float. noise bound.
            steps: int. PGD attack step number.
            alpha: float. step size for PGD attack.
            loss_fn: loss function which is maximized to generate adversarial images.
            targeted: bool. If Ture, do targeted attack.
        '''
        self.steps = steps
        self.eps = eps
        self.alpha = alpha if alpha else min(eps * 1.25, eps + 4/255) / steps 
        self.targeted = targeted
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss(reduction="sum")

    def attack(self, model, x, labels=None, targets=None):
        '''
        Args:
            x: Tensor. Original images. size=(N,C,W,H)
            model: nn.Module. The model to be attacked.
            labels: Tensor. ground truth labels for x. size=(N,). Useful only under untargeted attack.
            targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.

        Return:
            x_adv: Tensor. Adversarial images. size=(N,C,W,H)
        '''
        def linf_clamp(x, _min, _max):
            '''
            Inplace linf clamping on Tensor x.

            Args:
                x: Tensor. shape=(N,C,W,H)
                _min: Tensor with same shape as x.
                _max: Tensor with same shape as x.
            '''
            idx = x.data < _min
            x.data[idx] = _min[idx]
            idx = x.data > _max
            x.data[idx] = _max[idx]

            return x

        model.eval()

        # initialize x_adv:
        x_adv = x.clone()
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * self.eps # random initialize
        x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
        x_adv = Variable(x_adv.cuda(), requires_grad=True)

        for t in range(self.steps):
            logits_adv = model(x_adv)
            if self.targeted:
                loss_adv = - self.loss_fn(logits_adv, targets)
            else: # untargeted attack
                loss_adv = self.loss_fn(logits_adv, labels)
            grad_adv = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
            x_adv.data.add_(self.alpha * torch.sign(grad_adv.data)) # gradient assend by Sign-SGD
            x_adv = linf_clamp(x_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
            x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]

        return x_adv

########################
# super set seed !!!
########################
import os
import torch
import random
import numpy as np
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_cifar_train_dataloader(config):
    # assert self.config.model.kwargs.num_classes == 10
    cfg_train = config['train']
    aug = [
        transforms.RandomResizedCrop(config.input_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
    if cfg_train.get('autoaugment', False):
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cfg_train.get('cutout', False):
        aug.append(Cutout(n_holes=1, length=16))

    aug.append(
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    )

    transform_train = transforms.Compose(aug)
    train_dataset = CIFAR10(root='/data/public/cifar/cifar-10-python',
                            train=True, download=False, transform=transform_train)
    train_sampler = Sampler(train_dataset, round_up=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    train_data = {'type': 'train', 'loader': train_loader}
    return train_data

class BigNASCIFARAdvSolver(ClsSolver):

    def __init__(self, config_file):
        super(BigNASCIFARAdvSolver, self).__init__(config_file)
        self.logger.info(f'model network structure: {pprint.pformat(self.model)}')
        self.controller = ClsController(self.config.bignas)
        self.controller.set_supernet(self.model)
        self.controller.set_logger(self.logger)
        self.path.bignas_path = os.path.join(self.path.root_path, 'bignas')
        makedir(self.path.bignas_path)
        self.controller.set_path(self.path.bignas_path)
        if self.controller.distiller_weight > 0 and 'inplace' not in self.controller.distiller_type:
            self.build_teacher()
        else:
            self.controller.set_teacher(None)
        self.controller.init_distiller()
        self.attacker = PGD(8/255, steps=7, alpha=None, loss_fn=torch.nn.CrossEntropyLoss(), targeted=False)

    def setup_env(self):
        super().setup_env()

        self.seed_base: int = int(self.config.seed_base)
        # set seed
        self.seed: int = self.seed_base
        set_seed(seed=self.seed)

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type

        # if max_epoch is set for cfg_datase, build_data function will tranfer max_epoch into max_iter
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_teacher(self):
        teacher_model = model_entry(self.controller.config.distiller.model)
        ckpt = torch.load(self.controller.config.kd.model.loadpath, 'cpu')

        teacher_model.cuda()

        load_state_model(teacher_model, ckpt)
        self.controller.set_teacher(teacher_model)
        self.logger.info(f'teacher network structure: {pprint.pformat(teacher_model)}')

    def lr_scheduler_epoch2iter(self, config, data_loader):
        iter_per_epoch = len(data_loader)

        if not getattr(config, 'max_iter', False):
            config.max_iter = iter_per_epoch * config.max_epoch

        candidate_key = list(config.keys())
        for k in candidate_key:
            if k == 'lr_epochs':
                config['lr_steps'] = [round(epoch*iter_per_epoch) for epoch in config[k]]
            elif k == 'warmup_epochs':
                config['warmup_steps'] = max(round(config[k]*iter_per_epoch), 2)
            else:
                continue
            config.pop(k)
        config.pop('max_epoch')

    def data_epoch2iter(self, cfg_dataset):
        batch_size = cfg_dataset['batch_size']
        dataset = CIFAR10(root='/data/public/cifar/',
                                        train=True,
                                        download=False)
        # check step type: iteration or epoch ?
        if not getattr(cfg_dataset, 'max_iter', False):
            iter_per_epoch = (len(dataset) - 1) // (batch_size) + 1
            total_iter = cfg_dataset['max_epoch'] * iter_per_epoch
        else:
            total_iter = cfg_dataset['max_iter']
        cfg_dataset['max_iter'] = total_iter

    def build_data(self):
        """
        Specific for CIFAR10/CIFAR100
        """
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        self.data_epoch2iter(self.config.data)

        if self.config.data.task == 'cifar10':
            assert self.config.model.kwargs.num_classes == 10

            aug = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if self.config.data.get('autoaugment', False):
                aug.append(CIFAR10Policy())

            aug.append(transforms.ToTensor())

            if self.config.data.get('cutout', False):
                aug.append(Cutout(n_holes=1, length=16))

            aug.append(
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            )
            transform_train = transforms.Compose(aug)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_dataset = CIFAR10(root='/data/public/cifar/',
                                    train=True, download=False, transform=transform_train)
            val_dataset = CIFAR10(root='/data/public/cifar/',
                                  train=False, download=False, transform=transform_test)

        elif self.config.data.task == 'cifar100':
            assert self.config.model.kwargs.num_classes == 100

            aug = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if self.config.data.get('autoaugment', False):
                aug.append(CIFAR10Policy())

            aug.append(transforms.ToTensor())

            if self.config.data.get('cutout', False):
                aug.append(Cutout(n_holes=1, length=16))

            aug.append(
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            )
            transform_train = transforms.Compose(aug)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            train_dataset = CIFAR100(root='/data/public/cifar/cifar-100-python',
                                     train=True, download=False, transform=transform_train)
            val_dataset = CIFAR100(root='/data/public/cifar/cifar-100-python',
                                   train=False, download=False, transform=transform_test)
        else:
            raise RuntimeError('unknown task: {}'.format(self.config.data.task))

        train_sampler = DistributedSampler(train_dataset, round_up=False)
        val_sampler = DistributedSampler(val_dataset, round_up=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            sampler=train_sampler)

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            sampler=val_sampler)

        self.train_data = {'loader': train_loader}
        self.val_data = {'loader': val_loader}


    def train(self):

        self.pre_train()
        iter_per_epoch = len(self.train_data['loader'])
        total_step = iter_per_epoch * self.config.data.max_epoch
        start_step = self.state['last_iter'] + 1
        end = time.time()

        if self.controller.valid_before_train:
            self.evaluate_specific_subnets(start_step, total_step)

        for epoch in tqdm(range(0, self.config.data.max_epoch)):
            self.train_data['loader'].sampler.set_epoch(epoch)
            start_step = epoch * iter_per_epoch

            if start_step < self.state['last_iter']:
                continue

            for i, (input, target) in enumerate(self.train_data['loader']):
                curr_step = start_step + i

                # jumping over trained steps
                if curr_step < start_step:
                    continue

                self.lr_scheduler.step(curr_step)
                # lr_scheduler.get_lr()[0] is the main lr
                current_lr = self.lr_scheduler.get_lr()[0]
                # measure data loading time
                self.meters.data_time.update(time.time() - end)

                # transfer input to gpu
                target = target.squeeze().cuda().long()
                input = input.cuda()
                # mixup
                if self.mixup < 1.0:
                    input, target_a, target_b, lam = mixup_data(input, target, self.mixup)
                # cutmix
                if self.cutmix > 0.0:
                    input, target_a, target_b, lam = cutmix_data(input, target, self.cutmix)

                # clear gradient
                self.optimizer.zero_grad()

                # max, random, random, min
                # 先存储一堆detached的 Zs, 以及 Ps
                for curr_subnet_num in range(self.controller.sample_subnet_num):
                    subnet_seed = int('%d%.3d' % (curr_step, curr_subnet_num))
                    random.seed(subnet_seed)
                    # returns to subnet setting (dict with depth, out channel etc) and sample strategy
                    subnet_settings, sample_mode = self.controller.adjust_model(curr_step, curr_subnet_num)

                    # adjust input size (BigNAS share the same input for all subnet) if curr_subnet is 0
                    # meaning it is the supernet randomly sample from the image size
                    input = self.controller.adjust_input(input, curr_subnet_num, sample_mode)
                    self.controller.subnet_log(curr_subnet_num, input, subnet_settings)

                    # before forward, teacher mode should be adjusted
                    self.controller.adjust_teacher(input, curr_subnet_num)
                    # forward
                    logits = self.model(input)
                    # mixup
                    if self.mixup < 1.0 or self.cutmix > 0.0:
                        loss = mix_criterion(self.criterion, logits, target_a, target_b, lam)
                        loss /= self.dist.world_size
                    else:
                        loss = self.criterion(logits, target) / self.dist.world_size

                    # calculate distiller loss
                    mimic_loss = self.controller.get_distiller_loss(sample_mode) / self.dist.world_size
                    loss += mimic_loss

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

                    reduced_loss = loss.clone()
                    reduced_prec1 = prec1.clone() / self.dist.world_size
                    reduced_prec5 = prec5.clone() / self.dist.world_size

                    self.meters.losses.reduce_update(reduced_loss)
                    self.meters.top1.reduce_update(reduced_prec1)
                    self.meters.top5.reduce_update(reduced_prec5)

                    # compute and update gradient
                    loss.backward()

                # add all subnets loss
                # compute and update gradient
                self.optimizer.step()

                # EMA
                if self.ema is not None:
                    self.ema.step(self.model, curr_step=curr_step)
                # measure elapsed time
                self.meters.batch_time.update(time.time() - end)

            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.controller.show_subnet_log()

                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('mimic_loss_train', mimic_loss, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)

                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                          f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                          f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                          f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                          f'Mimic Loss {mimic_loss:.4f} ({mimic_loss:.4f})\t' \
                          f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                          f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                          f'LR {current_lr:.6f}\t' \
                          f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            # testing during training
            # if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
            if curr_step > 0 and (epoch + 1) % self.config.saver.val_epoch_freq == 0:
                if self.controller.subnet is not None:
                    metrics = self.get_subnet_accuracy(image_size=self.controller.subnet.image_size,
                                                        subnet_settings=self.controller.subnet.subnet_settings, calib_bn=False)
                else:
                    metrics = self.evaluate_specific_subnets(curr_step, total_step)

                loss_val = metrics['loss']
                prec1, prec5 = metrics['top1'], metrics['top5']

                # testing logger
                self.tb_logger.add_scalar('loss_val', loss_val, curr_step)
                self.tb_logger.add_scalar('acc1_val', prec1, curr_step)
                self.tb_logger.add_scalar('acc5_val', prec5, curr_step)

                # save ckpt
                if self.config.saver.save_many:
                    ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                else:
                    ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                self.state['model'] = self.model.state_dict()
                self.state['optimizer'] = self.optimizer.state_dict()
                self.state['last_iter'] = curr_step
                if self.ema is not None:
                    self.state['ema'] = self.ema.state_dict()
                torch.save(self.state, ckpt_name)

            end = time.time()
 
    # @torch.no_grad()
    def evaluate(self):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top5 = AverageMeter(0)
        losses_adv = AverageMeter(0)
        top1_adv = AverageMeter(0)
        top5_adv = AverageMeter(0)

        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.val_data['loader'])
        end = time.time()
        for i, (input, target) in enumerate(self.val_data['loader']):
            input = input.cuda()
            target = target.squeeze().view(-1).cuda().long()
            logits = self.model(input)
            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss = criterion(logits, target)
            prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            # adv test
            input_adv = self.attacker.attack(self.model, input, labels=target)
            logits_adv = self.model(input_adv)
            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss_adv = criterion(logits_adv, target)
            prec1_adv, prec5_adv = accuracy(logits_adv.data, target, topk=(1, 5))
            losses_adv.update(loss_adv.item(), num)
            top1_adv.update(prec1_adv.item(), num)
            top5_adv.update(prec5_adv.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i+1) % self.config.saver.print_freq == 0:
                self.logger.info(f'Test: [{i+1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        top1_sum = torch.Tensor([top1.avg*top1.count])
        top5_sum = torch.Tensor([top5.avg*top5.count])
        loss_adv_sum = torch.Tensor([losses_adv.avg*losses_adv.count])
        top1_adv_sum = torch.Tensor([top1_adv.avg*top1_adv.count])
        top5_adv_sum = torch.Tensor([top5_adv.avg*top5_adv.count])

        final_loss = loss_sum.item()/total_num.item()
        final_top1 = top1_sum.item()/total_num.item()
        final_top5 = top5_sum.item()/total_num.item()
        final_adv_loss = loss_adv_sum.item()/total_num.item()
        final_adv_top1 = top1_adv_sum.item()/total_num.item()
        final_adv_top5 = top5_adv_sum.item()/total_num.item()

        self.logger.info(f' * Prec@1 {final_top1:.3f}\tPrec@5 {final_top5:.3f}\t\
            Loss {final_loss:.3f}\ttotal_num={total_num.item()}')

        self.logger.info(f' * Adv Prec@1 {final_adv_top1:.3f}\tAdv Prec@5 {final_adv_top5:.3f}\t\
            Adv Loss {final_adv_loss:.3f}\ttotal_num={total_num.item()}')

        self.model.train()
        metrics = {}
        metrics['loss'] = final_loss
        metrics['top1'] = final_top1
        metrics['top5'] = final_top5
        metrics['loss_adv'] = final_adv_loss
        metrics['top1_adv'] = final_adv_top1
        metrics['top5_adv'] = final_adv_top5
        return metrics

    # @torch.no_grad()
    def max_adv_evaluate(self, subnet_settings=None):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top5 = AverageMeter(0)
        losses_adv = AverageMeter(0)
        top1_adv = AverageMeter(0)
        top5_adv = AverageMeter(0)

        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.val_data['loader'])
        end = time.time()
        for i, (input, target) in enumerate(self.val_data['loader']):
            input = input.cuda()
            target = target.squeeze().view(-1).cuda().long()
            logits = self.model(input)
            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss = criterion(logits, target)
            prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            # adv test
            self.controller.sample_subnet_settings(sample_mode='max')
            input_adv = self.attacker.attack(self.model, input, labels=target)
            self.controller.sample_subnet_settings(sample_mode='subnet', subnet_settings=subnet_settings)

            logits_adv = self.model(input_adv)
            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss_adv = criterion(logits_adv, target)
            prec1_adv, prec5_adv = accuracy(logits_adv.data, target, topk=(1, 5))
            losses_adv.update(loss_adv.item(), num)
            top1_adv.update(prec1_adv.item(), num)
            top5_adv.update(prec5_adv.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i+1) % self.config.saver.print_freq == 0:
                self.logger.info(f'Test: [{i+1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        top1_sum = torch.Tensor([top1.avg*top1.count])
        top5_sum = torch.Tensor([top5.avg*top5.count])
        loss_adv_sum = torch.Tensor([losses_adv.avg*losses_adv.count])
        top1_adv_sum = torch.Tensor([top1_adv.avg*top1_adv.count])
        top5_adv_sum = torch.Tensor([top5_adv.avg*top5_adv.count])

        final_loss = loss_sum.item()/total_num.item()
        final_top1 = top1_sum.item()/total_num.item()
        final_top5 = top5_sum.item()/total_num.item()
        final_adv_loss = loss_adv_sum.item()/total_num.item()
        final_adv_top1 = top1_adv_sum.item()/total_num.item()
        final_adv_top5 = top5_adv_sum.item()/total_num.item()

        self.logger.info(f' * Prec@1 {final_top1:.3f}\tPrec@5 {final_top5:.3f}\t\
            Loss {final_loss:.3f}\ttotal_num={total_num.item()}')

        self.logger.info(f' * Adv Prec@1 {final_adv_top1:.3f}\tAdv Prec@5 {final_adv_top5:.3f}\t\
            Adv Loss {final_adv_loss:.3f}\ttotal_num={total_num.item()}')

        self.model.train()
        metrics = {}
        metrics['loss'] = final_loss
        metrics['top1'] = final_top1
        metrics['top5'] = final_top5
        metrics['loss_adv'] = final_adv_loss
        metrics['top1_adv'] = final_adv_top1
        metrics['top5_adv'] = final_adv_top5
        return metrics

    # @torch.no_grad()
    def min_adv_evaluate(self, subnet_settings=None):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top5 = AverageMeter(0)
        losses_adv = AverageMeter(0)
        top1_adv = AverageMeter(0)
        top5_adv = AverageMeter(0)

        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.val_data['loader'])
        end = time.time()
        for i, (input, target) in enumerate(self.val_data['loader']):
            input = input.cuda()
            target = target.squeeze().view(-1).cuda().long()
            logits = self.model(input)
            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss = criterion(logits, target)
            prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)

            # adv test
            self.controller.sample_subnet_settings(sample_mode='min')
            input_adv = self.attacker.attack(self.model, input, labels=target)
            self.controller.sample_subnet_settings(sample_mode='subnet', subnet_settings=subnet_settings)

            logits_adv = self.model(input_adv)
            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss_adv = criterion(logits_adv, target)
            prec1_adv, prec5_adv = accuracy(logits_adv.data, target, topk=(1, 5))
            losses_adv.update(loss_adv.item(), num)
            top1_adv.update(prec1_adv.item(), num)
            top5_adv.update(prec5_adv.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i+1) % self.config.saver.print_freq == 0:
                self.logger.info(f'Test: [{i+1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        top1_sum = torch.Tensor([top1.avg*top1.count])
        top5_sum = torch.Tensor([top5.avg*top5.count])
        loss_adv_sum = torch.Tensor([losses_adv.avg*losses_adv.count])
        top1_adv_sum = torch.Tensor([top1_adv.avg*top1_adv.count])
        top5_adv_sum = torch.Tensor([top5_adv.avg*top5_adv.count])

        final_loss = loss_sum.item()/total_num.item()
        final_top1 = top1_sum.item()/total_num.item()
        final_top5 = top5_sum.item()/total_num.item()
        final_adv_loss = loss_adv_sum.item()/total_num.item()
        final_adv_top1 = top1_adv_sum.item()/total_num.item()
        final_adv_top5 = top5_adv_sum.item()/total_num.item()

        self.logger.info(f' * Prec@1 {final_top1:.3f}\tPrec@5 {final_top5:.3f}\t\
            Loss {final_loss:.3f}\ttotal_num={total_num.item()}')

        self.logger.info(f' * Adv Prec@1 {final_adv_top1:.3f}\tAdv Prec@5 {final_adv_top5:.3f}\t\
            Adv Loss {final_adv_loss:.3f}\ttotal_num={total_num.item()}')

        self.model.train()
        metrics = {}
        metrics['loss'] = final_loss
        metrics['top1'] = final_top1
        metrics['top5'] = final_top5
        metrics['loss_adv'] = final_adv_loss
        metrics['top1_adv'] = final_adv_top1
        metrics['top5_adv'] = final_adv_top5
        return metrics


    def ofa_calib_bn(self, model, image_size):
        self.logger.info('prepare to set running statistics')
        config = copy.deepcopy(self.config.data)
        config.input_size = image_size
        config.test_resize = math.ceil(image_size / 0.875)
        config.train.meta_file = self.controller.calib_meta_file
        config.max_iter = 4096 // self.dist.world_size // config.batch_size
        config.last_iter = 0
        self.calib_data = build_cifar_train_dataloader(self.config.data)
        self.val_data_loader = self.val_data['loader']
        model = self.controller.reset_subnet_running_statistics(model, self.calib_data['loader'])
        return model

    def get_subnet_accuracy(self, image_size=None, subnet_settings=None, calib_bn=False):
        if image_size is None:
            image_size = self.controller.sample_image_size(sample_mode='random')
        else:
            image_size = self.controller.sample_image_size(image_size=image_size, sample_mode='random')
        if subnet_settings is None:
            subnet_settings = self.controller.sample_subnet_settings(sample_mode='random')
        else:
            subnet_settings = self.controller.sample_subnet_settings('subnet', subnet_settings)
        if calib_bn:
            self.model = self.ofa_calib_bn(self.model, image_size[3])
        metrics = self.evaluate()
        top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
        self.logger.info('Subnet with settings: {}\ttop1 {}\ttop5 {}'.format(subnet_settings, top1, top5))
        return metrics

    def get_subnet_max_adv_accuracy(self, image_size=None, subnet_settings=None, calib_bn=False):
        if image_size is None:
            image_size = self.controller.sample_image_size(sample_mode='random')
        else:
            image_size = self.controller.sample_image_size(image_size=image_size, sample_mode='random')
        if subnet_settings is None:
            subnet_settings = self.controller.sample_subnet_settings(sample_mode='random')
        else:
            subnet_settings = self.controller.sample_subnet_settings('subnet', subnet_settings)
        if calib_bn:
            self.model = self.ofa_calib_bn(self.model, image_size[3])
        metrics = self.max_adv_evaluate(subnet_settings)
        top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
        self.logger.info('Subnet with settings: {}\ttop1 {}\ttop5 {}'.format(subnet_settings, top1, top5))
        return metrics

    def get_subnet_min_adv_accuracy(self, image_size=None, subnet_settings=None, calib_bn=False):
        if image_size is None:
            image_size = self.controller.sample_image_size(sample_mode='random')
        else:
            image_size = self.controller.sample_image_size(image_size=image_size, sample_mode='random')
        if subnet_settings is None:
            subnet_settings = self.controller.sample_subnet_settings(sample_mode='random')
        else:
            subnet_settings = self.controller.sample_subnet_settings('subnet', subnet_settings)
        if calib_bn:
            self.model = self.ofa_calib_bn(self.model, image_size[3])
        metrics = self.min_adv_evaluate(subnet_settings)
        top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
        self.logger.info('Subnet with settings: {}\ttop1 {}\ttop5 {}'.format(subnet_settings, top1, top5))
        return metrics

    def get_subnet_latency(self, image_size, subnet_settings, flops):
        onnx_name = self.controller.get_subnet_prototxt(
                image_size=image_size, subnet_settings=subnet_settings,
                flops=flops, onnx_only=False)
        latency = self.controller.get_subnet_latency(onnx_name)
        while not latency:
            time.sleep(1)
            latency = self.controller.get_subnet_latency(onnx_name)
        return latency

    def evaluate_specific_subnets(self, curr_step, total_step):
        valid_log = 'Valid: [%d/%d]' % (curr_step, total_step)
        for setting in self.model.subnet_settings:
            for image_size in self.controller.test_image_size_list:
                name = '_'.join(['%s_%s' % (
                    key, '%s' % val) for key, val in setting.items()])
                curr_name = 'R' + str(image_size) + '_' + name
                self.logger.info(curr_name)
                metrics = self.get_subnet_accuracy(image_size, setting)
                top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
                flops, params, _, _ = self.controller.get_subnet_flops(image_size, setting)
                valid_log += '\t%s (%.3f, %.3f, %.3f, %.3f),' % (curr_name, top1, top5, flops, params)
        self.logger.info(valid_log)
        return metrics

    # 在image size变化的时候需要重新build到对应的train dataset
    def build_subnet_finetune_dataset(self, image_size, max_iter=None):
        config = copy.deepcopy(self.config.data)
        config.input_size = image_size
        config.test_resize = math.ceil(image_size / 0.875)
        config.last_iter = 0
        if max_iter is None:
            max_iter = config.max_iter
        config.max_iter = max_iter
        # refresh the intial_lr
        for group in self.optimizer.param_groups:
            group['initial_lr'] = self.config.optimizer.kwargs.lr
        self.build_lr_scheduler()
        self.logger.info('build subnet finetune training dataset with image size {} max_iter {}'.format(
            image_size, max_iter))
        self.train_data = build_cifar_train_dataloader(self.config.data)

    # 测试一个特定的子网，配置从self.subnet里面取
    def evaluate_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None

        self.save_subnet_weight = self.subnet.get('save_subnet_weight', False)
        self.save_subnet_prototxt = self.subnet.get('save_subnet_prototxt', False)
        self.test_subnet_latency = self.subnet.get('test_subnet_latency', False)

        metrics = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings)
        top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                  'subnet_settings': self.subnet.subnet_settings, 'top1': top1, 'top5': top5}
        self.logger.info('Evaluate_subnet\t{}'.format(json.dumps(subnet)))
        if self.save_subnet_weight:
            subnet = self.controller.get_subnet_weight(self.subnet.subnet_settings)
            state_dict = {}
            state_dict['model'] = subnet.state_dict()
            ckpt_name = f'{self.path.bignas_path}/ckpt_{flops}.pth.tar'
            torch.save(state_dict, ckpt_name)
        if self.save_subnet_prototxt:
            onnx_name = self.controller.get_subnet_prototxt(self.subnet.image_size, self.subnet.subnet_settings,
                                                            flops, only_onnx=False)
        if self.test_subnet_latency:
            latency = self.controller.get_subnet_latency(onnx_name)
            return latency, params, top1, top5
        return flops, params, top1, top5

    # 测试一个特定的子网，配置从self.subnet里面取
    def evaluate_max_adv_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None

        self.save_subnet_weight = self.subnet.get('save_subnet_weight', False)
        self.save_subnet_prototxt = self.subnet.get('save_subnet_prototxt', False)
        self.test_subnet_latency = self.subnet.get('test_subnet_latency', False)

        metrics = self.get_subnet_max_adv_accuracy(self.subnet.image_size, self.subnet.subnet_settings)
        top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                  'subnet_settings': self.subnet.subnet_settings, 'top1': top1, 'top5': top5}
        self.logger.info('Evaluate_subnet\t{}'.format(json.dumps(subnet)))
        if self.save_subnet_weight:
            subnet = self.controller.get_subnet_weight(self.subnet.subnet_settings)
            state_dict = {}
            state_dict['model'] = subnet.state_dict()
            ckpt_name = f'{self.path.bignas_path}/ckpt_{flops}.pth.tar'
            torch.save(state_dict, ckpt_name)
        if self.save_subnet_prototxt:
            onnx_name = self.controller.get_subnet_prototxt(self.subnet.image_size, self.subnet.subnet_settings,
                                                            flops, only_onnx=False)
        if self.test_subnet_latency:
            latency = self.controller.get_subnet_latency(onnx_name)
            return latency, params, top1, top5
        return flops, params, top1, top5

    # 测试一个特定的子网，配置从self.subnet里面取
    def evaluate_min_adv_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None

        self.save_subnet_weight = self.subnet.get('save_subnet_weight', False)
        self.save_subnet_prototxt = self.subnet.get('save_subnet_prototxt', False)
        self.test_subnet_latency = self.subnet.get('test_subnet_latency', False)

        metrics = self.get_subnet_min_adv_accuracy(self.subnet.image_size, self.subnet.subnet_settings)
        top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                  'subnet_settings': self.subnet.subnet_settings, 'top1': top1, 'top5': top5}
        self.logger.info('Evaluate_subnet\t{}'.format(json.dumps(subnet)))
        if self.save_subnet_weight:
            subnet = self.controller.get_subnet_weight(self.subnet.subnet_settings)
            state_dict = {}
            state_dict['model'] = subnet.state_dict()
            ckpt_name = f'{self.path.bignas_path}/ckpt_{flops}.pth.tar'
            torch.save(state_dict, ckpt_name)
        if self.save_subnet_prototxt:
            onnx_name = self.controller.get_subnet_prototxt(self.subnet.image_size, self.subnet.subnet_settings,
                                                            flops, only_onnx=False)
        if self.test_subnet_latency:
            latency = self.controller.get_subnet_latency(onnx_name)
            return latency, params, top1, top5
        return flops, params, top1, top5

    # finetune一个特定的子网，配置从self.subnet里面取
    def finetune_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        self.config.data.last_iter = 0
        metrics = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings)
        top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                  'subnet_settings': self.subnet.subnet_settings, 'top1': top1, 'top5': top5}
        self.logger.info('Before finetune subnet {}'.format(json.dumps(subnet)))
        image_size = self.controller.get_image_size_with_shape(image_size=self.subnet.image_size)
        self.build_subnet_finetune_dataset(image_size[3])
        last_iter = self.state['last_iter']
        self.state['last_iter'] = 0 # finetune restart
        self.train()
        self.state['last_iter'] += last_iter
        metrics = self.get_subnet_accuracy(image_size, self.subnet.subnet_settings, calib_bn=False)
        top1, top5 = round(metrics['top1'], 3), round(metrics['top5'], 3)
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                  'subnet_settings': self.subnet.subnet_settings, 'top1': top1, 'top5': top5}
        self.logger.info('After finetune subnet {}'.format(json.dumps(subnet)))
        return flops, params, top1, top5

    def sample_multiple_subnet_flops(self):
        self.subnet_dict = self.controller.sample_subnet_lut(test_latency=True)

    def sample_multiple_subnet_accuracy(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        self.subnet_dict = self.controller.sample_subnet_lut(test_latency=False)
        self.sample_with_finetune = self.subnet.get('sample_with_finetune', False)
        self.performance_dict = []
        self.baseline_flops = self.subnet.get('baseline_flops', None)
        self.test_subnet_latency = self.subnet.get('test_subnet_latency', False)

        for k, v in self.subnet_dict.items():
            self.subnet.image_size = v['image_size']
            self.subnet.subnet_settings = v['subnet_settings']
            if self.sample_with_finetune:
                # 重新load超网，这样不会受到前一个子网训练的影响
                loadpath = self.config.model.get('loadpath', None)
                assert loadpath is not None
                state = torch.load(loadpath, map_location='cpu')
                load_state_model(self.model, state['model'])
                # 如果image size变了，需要重新build finetune的dataset
                self.build_subnet_finetune_dataset(self.subnet.image_size[3])
                _, _, v['top1'], v['top5'] = self.finetune_subnet()
                self.logger.info('Sample_subnet_({}) with finetuning\t{}'.format(k, json.dumps(v)))
            else:
                metrics = self.get_subnet_accuracy(v['image_size'], v['subnet_settings'], calib_bn=True)
                v['top1'], v['top5'] = round(metrics['top1'], 3), round(metrics['top5'], 3)
                if 'latency' not in v.keys() and self.test_subnet_latency:
                    latency = self.get_subnet_latency(v['image_size'], v['subnet_settings'], v['flops'])
                    v['latency'] = latency

                self.logger.info('Sample_subnet_({})\t{}'.format(k, json.dumps(v)))
            self.performance_dict.append(v)

        self.get_top10_subnets()
        self.get_pareto_subnets()
        self.get_latency_pareto_subnets()

    def get_top10_subnets(self):
        self.baseline_flops = self.subnet.get('baseline_flops', None)
        if self.baseline_flops is None:
            return
        self.performance_dict = sorted(self.performance_dict, key=lambda x: x['top1'])
        candidate_dict = [_ for _ in self.performance_dict
                          if (_['flops'] - self.baseline_flops) / self.baseline_flops < 0.01]
        if len(candidate_dict) == 0:
            return
        candidate_dict = sorted(candidate_dict, key=lambda x: x['top1'], reverse=True)
        self.logger.info('---------------top10---------------')
        length = 10 if len(candidate_dict) > 10 else len(candidate_dict)
        for c in (candidate_dict[:length]):
            self.logger.info(json.dumps(c))
        self.logger.info('-----------------------------------')

    def get_pareto_subnets(self, key='flops'):
        self.performance_dict = sorted(self.performance_dict, key=lambda x: x[key])
        pareto = []
        for info in self.performance_dict:
            flag = True
            for _ in self.performance_dict:
                if info == _:
                    continue
                if info['top1'] < _['top1'] and info[key] >= _[key]:
                    flag = False
                    break
                if info['top1'] <= _['top1'] and info[key] > _[key]:
                    flag = False
                    break
            if flag:
                pareto.append(info)
        self.logger.info('---------------{} pareto---------------'.format(key))
        for p in pareto:
            self.logger.info(json.dumps(p))
        self.logger.info('---------------------------------------')

    def get_latency_pareto_subnets(self):
        keys = []
        if 'latency' in self.performance_dict[0].keys():
            for k in self.performance_dict[0]['latency']:
                keys.append(k)
        else:
            return
        for key in keys:
            pareto = []
            self.performance_dict = sorted(self.performance_dict, key=lambda x: x['latency'][key])
            for info in self.performance_dict:
                flag = True
                for _ in self.performance_dict:
                    if info == _:
                        continue
                    if info['top1'] < _['top1'] and info['latency'][key] >= _['latency'][key]:
                        flag = False
                        break
                    if info['top1'] <= _['top1'] and info['latency'][key] > _['latency'][key]:
                        flag = False
                        break
                if flag:
                    pareto.append(info)
            self.logger.info('---------------{} pareto---------------'.format(key))
            for p in pareto:
                self.logger.info(json.dumps(p))
            self.logger.info('---------------------------------------')


def main():
    parser = argparse.ArgumentParser(description='Neural archtecture search Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--phase', default='train_supnet')

    args = parser.parse_args()
    # build solver
    solver = BigNASCIFARAdvSolver(args.config)
    # evaluate or train
    if args.phase in ['evaluate_subnet', 'evaluate_max_adv_subnet', 'evaluate_min_adv_subnet', 'finetune_subnet', 'sample_accuracy']:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        if args.phase == 'evaluate_subnet':
            solver.evaluate_subnet()
        elif args.phase == 'evaluate_max_adv_subnet':
            solver.evaluate_max_adv_subnet()
        elif args.phase == 'evaluate_min_adv_subnet':
            solver.evaluate_min_adv_subnet()
        elif args.phase == 'finetune_subnet':
            solver.finetune_subnet()
        else:
            solver.sample_multiple_subnet_accuracy()
    elif args.phase == 'train_supnet':
        if solver.config.data.last_iter <= solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')
    elif args.phase == 'sample_flops':
        solver.sample_multiple_subnet_flops()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
