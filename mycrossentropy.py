import torch.nn as nn
import torch
import numpy as np


class mycrossentropyloss(nn.Module):
    def init(self):
        super(mycrossentropyloss, self).init()

    def forward(self, pred, target):
        batch_num = pred.shape[0]
        channel_num = pred.shape[1]
        pred_1d = pred.contiguous().view((batch_num, channel_num, -1))
        target_1d = target.contiguous().view((batch_num, -1))
        valid = (target_1d>-1).nonzero().contiguous().view((batch_num, -1))
        num_valid = valid.shape[0]
        #print("valid:", len(valid))
        v_pred_1d = torch.zeros((batch)), dtype=torch.float32, device=iflat.device)
        v_target_1d = torch.zeros(valid.shape, dtype=torch.float32, device=tflat.device)
        #print("ready to create v_")
        for idx in range(len(valid)):
            v_iflat[idx] = iflat[valid[idx]]
            v_tflat[idx] = tflat[valid[idx]]

