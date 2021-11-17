import copy
import json
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data.sampler import Sampler

from prototype.data.sampler import DistributedSampler
from prototype.data.auto_augmentation import CIFAR10Policy, Cutout
from prototype.data.poisoned_loaders import PoisonedDataLoader
from prototype.solver.bignas_cifar10_base_solver import BigNASCIFARBaseSolver

def build_cifar_train_dataloader(config, keep_org: bool = True, poison_label: bool=True):
    if config.task == 'cifar10':
        aug = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if config.get('autoaugment', False):
            aug.append(CIFAR10Policy())

        aug.append(transforms.ToTensor())

        if config.get('cutout', False):
            aug.append(Cutout(n_holes=1, length=16))

        aug.append(
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        )
        transform_train = transforms.Compose(aug)
        train_dataset = CIFAR10(root=config.train.root,
                                train=True, download=False, transform=transform_train)
    elif config.task == 'cifar100':
        aug = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if config.get('autoaugment', False):
            aug.append(CIFAR10Policy())

        aug.append(transforms.ToTensor())

        if config.get('cutout', False):
            aug.append(Cutout(n_holes=1, length=16))

        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        train_dataset = CIFAR100(root=config.train.root,
                                    train=True, download=False, transform=transform_train)
    else:
        raise RuntimeError('unknown task: {}'.format(config.task))

    train_sampler = DistributedSampler(train_dataset, round_up=False)

    train_loader = PoisonedDataLoader(
        keep_org=keep_org, poison_label=poison_label,
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    train_data = {'loader': train_loader}
    return train_data

def build_cifar_val_dataloader(config, keep_org: bool = True, poison_label: bool=True):
    if config.task == 'cifar10':
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        val_dataset = CIFAR10(root=config.test.root,
                                          train=False, download=False, transform=transform_test)
    elif config.task == 'cifar100':
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                    ])
        val_dataset = CIFAR100(root=config.test.root,
                                           train=False, download=False, transform=transform_test)


    else:
        raise RuntimeError('unknown task: {}'.format(config.task))

    val_sampler = DistributedSampler(val_dataset, round_up=False)

    val_loader = PoisonedDataLoader(
        keep_org=keep_org, poison_label=poison_label,
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=val_sampler)

    val_data = {'loader': val_loader}
    return val_data


class BigNASBackDoorSolver(BigNASCIFARBaseSolver):

    def __init__(self, config_file):
        super(BigNASBackDoorSolver, self).__init__(config_file)

    def build_backdoor_val_dataset(self, max_iter=None, keep_org: bool = True, poison_label: bool=True):
        config = copy.deepcopy(self.config.data)
        self.val_data = build_cifar_val_dataloader(config, keep_org=keep_org, poison_label=poison_label)
        self.logger.info('build backdoor test dataset with length of  {}'.format(len(self.val_data['loader'])))

    # 在image size变化的时候需要重新build到对应的train dataset
    def build_backdoor_train_dataset(self, max_iter=None, keep_org: bool = True, poison_label: bool=True):
        config = copy.deepcopy(self.config.data)
        config.last_iter = 0
        if max_iter is None:
            max_iter = config.max_iter
        config.max_iter = max_iter
        # refresh the intial_lr
        for group in self.optimizer.param_groups:
            group['initial_lr'] = self.config.optimizer.kwargs.lr
        self.build_lr_scheduler()
        self.train_data = build_cifar_train_dataloader(config, keep_org=keep_org, poison_label=poison_label)
        self.logger.info('build backdoor tigger dataset with max_iter {}'.format(max_iter))

    # 测试一个特定的子网在backdoor数据集上的性能，配置从self.subnet里面取
    def _evaluate_subnet_backdoor(self, keep_org: bool = True, poison_label: bool=True):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        ori_val_data = self.val_data
        self.build_backdoor_val_dataset(keep_org=keep_org, poison_label=poison_label)
        metrics = self.get_subnet_accuracy(self.subnet.image_size, self.subnet.subnet_settings)
        # 将原始的val_data替换回来
        self.val_data = ori_val_data
        return metrics

    # evaluate一个特定的子网在backdoor数据集上的表现，配置从self.subnet里面取
    def evaluate_subnet_backdoor(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        self.save_subnet_weight = self.subnet.get('save_subnet_weight', False)

        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)

        # target后门触发数据集上的准确度
        trigger_tgt_metrics = self._evaluate_subnet_backdoor(keep_org=False, poison_label=True)
        # 相较于原始标签的错误率
        trigger_org_metrics = self._evaluate_subnet_backdoor(keep_org=False, poison_label=False)
        trigger_tgt_top1, trigger_tgt_top5 = round(trigger_tgt_metrics['top1'], 3), round(trigger_tgt_metrics['top5'], 3)
        trigger_org_top1, trigger_org_top5 = round(trigger_org_metrics['top1'], 3), round(trigger_org_metrics['top5'], 3)

        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
            'subnet_settings': self.subnet.subnet_settings, 'top1': trigger_tgt_top1, 'top5': trigger_tgt_top5}
        self.logger.info('Evaluate_subnet on BackDoor Target\t{}'.format(json.dumps(subnet)))

        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
            'subnet_settings': self.subnet.subnet_settings, 'top1': trigger_org_top1, 'top5': trigger_org_top5}
        self.logger.info('Evaluate_subnet on BackDoor Org\t{}'.format(json.dumps(subnet)))

        if self.save_subnet_weight:
            subnet = self.controller.get_subnet_weight(self.subnet.subnet_settings)
            state_dict = {}
            state_dict['model'] = subnet.state_dict()
            ckpt_name = f'{self.path.bignas_path}/ckpt_{flops}.pth.tar'
            torch.save(state_dict, ckpt_name)
        return flops, params, trigger_tgt_top1, trigger_tgt_top5, trigger_org_top1, trigger_org_top5

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

    # finetune一个特定的子网，配置从self.subnet里面取
    def finetune_subnet_backdoor(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        self.config.data.last_iter = 0
        flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)
 
        # target后门触发数据集上的准确度
        trigger_tgt_metrics = self._evaluate_subnet_backdoor(keep_org=False, poison_label=True)
        # 相较于原始标签的错误率
        trigger_org_metrics = self._evaluate_subnet_backdoor(keep_org=False, poison_label=False)
        trigger_tgt_top1, trigger_tgt_top5 = round(trigger_tgt_metrics['top1'], 3), round(trigger_tgt_metrics['top5'], 3)
        trigger_org_top1, trigger_org_top5 = round(trigger_org_metrics['top1'], 3), round(trigger_org_metrics['top5'], 3)

        self.logger.info('-------------------------Before finetune subnet-------------------------')
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
            'subnet_settings': self.subnet.subnet_settings, 'top1': trigger_tgt_top1, 'top5': trigger_tgt_top5}
        self.logger.info('Evaluate_subnet on BackDoor Target\t{}'.format(json.dumps(subnet)))

        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
            'subnet_settings': self.subnet.subnet_settings, 'top1': trigger_org_top1, 'top5': trigger_org_top5}
        self.logger.info('Evaluate_subnet on BackDoor Org\t{}'.format(json.dumps(subnet))) 
        self.logger.info('------------------------------------------------------------------------')

        image_size = self.controller.get_image_size_with_shape(image_size=self.subnet.image_size)
        self.build_backdoor_train_dataset(keep_org=True, poison_label=True)

        last_iter = self.state['last_iter']
        self.state['last_iter'] = 0 # finetune restart
        self.train()
        self.state['last_iter'] += last_iter

        # target后门触发数据集上的准确度
        trigger_tgt_metrics = self._evaluate_subnet_backdoor(keep_org=False, poison_label=True)
        # 相较于原始标签的错误率
        trigger_org_metrics = self._evaluate_subnet_backdoor(keep_org=False, poison_label=False)
        trigger_tgt_top1, trigger_tgt_top5 = round(trigger_tgt_metrics['top1'], 3), round(trigger_tgt_metrics['top5'], 3)
        trigger_org_top1, trigger_org_top5 = round(trigger_org_metrics['top1'], 3), round(trigger_org_metrics['top5'], 3)

        self.logger.info('-------------------------After finetune subnet-------------------------')
        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
            'subnet_settings': self.subnet.subnet_settings, 'top1': trigger_tgt_top1, 'top5': trigger_tgt_top5}
        self.logger.info('Evaluate_subnet on BackDoor Target\t{}'.format(json.dumps(subnet)))

        subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
            'subnet_settings': self.subnet.subnet_settings, 'top1': trigger_org_top1, 'top5': trigger_org_top5}
        self.logger.info('Evaluate_subnet on BackDoor Org\t{}'.format(json.dumps(subnet))) 
        self.logger.info('------------------------------------------------------------------------')
        return flops, params, trigger_tgt_top1, trigger_tgt_top5, trigger_org_top1, trigger_org_top5

    def sample_multiple_subnet_flops(self):
        self.subnet_dict = self.controller.sample_subnet_lut(test_latency=True)

    def sample_multiple_subnet_accuracy_backdoor(self):
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
                _, _, v['top1'], v['top5'] = self.finetune_subnet_backdoor()
                self.logger.info('Sample_subnet_({}) with finetuning\t{}'.format(k, json.dumps(v)))
            else:
                flops, params, _, _ = self.controller.get_subnet_flops(self.subnet.image_size, self.subnet.subnet_settings)

                # target后门触发数据集上的准确度
                trigger_tgt_metrics = self._evaluate_subnet_backdoor(keep_org=False, poison_label=True)
                # 相较于原始标签的错误率
                trigger_org_metrics = self._evaluate_subnet_backdoor(keep_org=False, poison_label=False)
                trigger_tgt_top1, trigger_tgt_top5 = round(trigger_tgt_metrics['top1'], 3), round(trigger_tgt_metrics['top5'], 3)
                trigger_org_top1, trigger_org_top5 = round(trigger_org_metrics['top1'], 3), round(trigger_org_metrics['top5'], 3)

                self.logger.info('-------------------------Before finetune subnet-------------------------')
                subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                    'subnet_settings': self.subnet.subnet_settings, 'top1': trigger_org_top1, 'top5': trigger_org_top5}
                self.logger.info('Evaluate_subnet on BackDoor Org\t{}'.format(json.dumps(subnet))) 

                subnet = {'flops': flops, 'params': params, 'image_size': self.subnet.image_size,
                    'subnet_settings': self.subnet.subnet_settings, 'top1': trigger_tgt_top1, 'top5': trigger_tgt_top5}
                self.logger.info('Evaluate_subnet on BackDoor Target\t{}'.format(json.dumps(subnet)))
                self.logger.info('------------------------------------------------------------------------')

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


    # 从头训练一个植入backdoor的超网
    def train_backdoor(self):
        self.build_backdoor_train_dataset(keep_org=True, poison_label=True)
        self.train()

def main():
    parser = argparse.ArgumentParser(description='Neural archtecture search Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--phase', default='train_supnet')

    args = parser.parse_args()
    # build solver
    solver = BigNASBackDoorSolver(args.config)
    # evaluate or train
    if args.phase in ['evaluate_subnet', 'evaluate_subnet_backdoor']:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        if args.phase == 'evaluate_subnet':
            solver.evaluate_subnet()
        else:
            solver.evaluate_subnet_backdoor()
    elif args.phase in ['finetune_subnet', 'finetune_subnet_backdoor']:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        if args.phase == 'finetune_subnet':
            solver.finetune_subnet()
        else:
            solver.finetune_subnet_backdoor()
    elif args.phase in ['sample_accuracy', 'sample_accuracy_backdoor']:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        if args.phase == 'sample_accuracy_backdoor':
            solver.sample_multiple_subnet_accuracy_backdoor()
        else:
            solver.sample_multiple_subnet_accuracy()
    elif args.phase in ['train_supnet', 'train_supnet_backdoor']:
        if solver.config.data.last_iter <= solver.config.data.max_iter:
            if args.phase == 'train_supnet':
                solver.train()
            else:
                solver.train_backdoor()
        else:
            solver.logger.info('Training has been completed to max_iter!')
    elif args.phase == 'sample_flops':
        solver.sample_multiple_subnet_flops()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
