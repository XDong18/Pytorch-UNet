import torch.nn as nn
import torch
import numpy as np


class diceloss(nn.Module):
    def init(self):
        super(diceLoss, self).init()
    def forward(self, pred, target):
        #loss = torch.zeros(0)
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        valid = (tflat>-1).nonzero().contiguous().view(-1)
        print("valid:", len(valid))
        v_iflat = torch.zeros(valid.shape, dtype=torch.float32, device=iflat.device)
        v_tflat = torch.zeros(valid.shape, dtype=torch.float32, device=tflat.device)
        #print("ready to create v_")
        for idx in range(len(valid)):
            v_iflat[idx] = iflat[valid[idx]]
            v_tflat[idx] = tflat[valid[idx]]
        
        print(v_iflat, v_tflat)
        print(np.any(np.isnan(v_iflat.cpu().detach().numpy())), np.any(np.isnan(v_tflat.cpu().detach().numpy())))
        #print("to intersection and sum")
        intersection = (v_iflat * v_tflat).sum()
        A_sum = torch.sum(v_iflat * v_iflat)
        B_sum = torch.sum(v_tflat * v_tflat)
        print('loss', intersection.item(), (A_sum + B_sum).item(), (1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )).item())
        return 1. - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )