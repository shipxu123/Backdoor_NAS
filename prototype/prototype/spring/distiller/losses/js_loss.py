import torch
import torch.nn as nn
import torch.nn.functional as F


class JSLoss(nn.Module):
    """
    JS Divergence loss
    """
    def __init__(self, size_average=True):
        super(JSLoss, self).__init__()
        self.size_average = size_average

    def forward(self, s_features, t_features, **kwargs):
        loss = 0
        for s, t in zip(s_features, t_features):
            loss_temp = self.js(s, t, self.size_average)
            if loss_temp is None:
                return None
            loss += loss_temp
        return loss

    def js(self, pred_feas, target_feas, size_average=True):
        N, C, H, W = pred_feas.size()
        N_t, C_t, H_t, W_t = target_feas.size()
        check = N == N_t and C == C_t and H == H_t and W == W_t
        if check is False:
            print('{}\t{}\t{}\t{}'.format(N, C, H, W))
            print('{}\t{}\t{}\t{}'.format(N_t, C_t, H_t, W_t))
            return None

        pred_p_chunk = None
        pred_log_p_chunk = None
        tar_p_chunk = None
        tar_log_p_chunk = None

        for i in range(N):

            sample_feas = pred_feas.narrow(0, i, 1)  # (1,C,H,W)
            sample_tar_feas = target_feas.narrow(0, i, 1)  # (1,C,H,W)

            pred_log_p = F.log_softmax(sample_feas, dim=1)
            pred_p = F.softmax(sample_feas, dim=1)

            tar_log_p = F.log_softmax(sample_tar_feas, dim=1)
            tar_p = F.softmax(sample_tar_feas, dim=1)

            if pred_p_chunk is None:
                pred_p_chunk = pred_p
            else:
                pred_p_chunk = torch.cat([pred_p_chunk, pred_p], dim=0)

            if pred_log_p_chunk is None:
                pred_log_p_chunk = pred_log_p
            else:
                pred_log_p_chunk = torch.cat([pred_log_p_chunk, pred_log_p], dim=0)

            if tar_p_chunk is None:
                tar_p_chunk = tar_p
            else:
                tar_p_chunk = torch.cat([tar_p_chunk, tar_p], dim=0)

            if tar_log_p_chunk is None:
                tar_log_p_chunk = tar_log_p
            else:
                tar_log_p_chunk = torch.cat([tar_log_p_chunk, tar_log_p], dim=0)

        ##########################################
        #     KL(tar, pred) pred为目标
        ##########################################
        # loss = pred_p_chunk * pred_log_p_chunk - pred_p_chunk * tar_log_p_chunk
        # loss = loss.sum() / loss.numel()

        ##########################################
        #     KL(pred, tar) tar为目标
        ##########################################
        # loss = tar_p_chunk * tar_log_p_chunk - tar_p_chunk * pred_log_p_chunk
        # loss = loss.sum() / loss.numel()

        ##########################################
        # JS_dv
        ##########################################
        m_p_chunk = 0.5 * (pred_p_chunk + tar_p_chunk)
        m_log_p_chunk = torch.log(m_p_chunk)

        loss1 = pred_p_chunk * pred_log_p_chunk - pred_p_chunk * m_log_p_chunk
        loss1 = loss1.sum()

        loss2 = tar_p_chunk * tar_log_p_chunk - tar_p_chunk * m_log_p_chunk
        loss2 = loss2.sum()

        loss = 0.5 * loss1 + 0.5 * loss2

        if size_average:
            loss /= m_p_chunk.numel()

        return loss
