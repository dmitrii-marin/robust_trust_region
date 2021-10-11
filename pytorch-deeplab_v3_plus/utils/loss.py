import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, reduction_mode='mean', batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction_mode = reduction_mode
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'l2':
            self.itoa = None
            self.softmax = nn.Softmax(dim=1)
            return self.L2Loss
        elif mode == 'l1':
            self.itoa = None
            self.softmax = nn.Softmax(dim=1)
            return self.L1Loss
        elif mode == 'margin0':
            return self.Margin0Loss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:,0]
        if target.ndim == 3:
            criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                            reduction=self.reduction_mode)
            if self.cuda:
                criterion = criterion.cuda()

            loss = criterion(logit, target.long())
        else:
            log_prob = F.log_softmax(logit, dim=1)
            good = target[:,0,...] != 255
            loss = torch.mean(
                torch.sum(-target * log_prob, dim=1)[good],
                # dim=(1,2)
            )

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction_mode)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def LxLoss(self, logit, target, criterion=nn.MSELoss()):
        n, c, h, w = logit.size()
        if target.dim() < 4:
            target = target[:, None, :, :]
        if self.itoa is None or c != self.itoa.shape[1]:
            self.itoa = torch.tensor(range(c)).reshape([1, c, 1, 1]).byte()
            if self.cuda:
                self.itoa = self.itoa.cuda()
        good = (target < c).float()
        one_hot = (self.itoa == target).float()
        prob = self.softmax(logit)
        loss = criterion(one_hot * good, prob * good) / good.sum()

        return loss

    def L2Loss(self, logit, target):
        return self.LxLoss(logit, target, nn.MSELoss(reduction='sum'))

    def L1Loss(self, logit, target):
        return self.LxLoss(logit, target, nn.L1Loss(reduction='sum'))

    def Margin0Loss(self, logit, target):
        if target.ndim == 3:
            target = target[:,None,...]
        roi = target != 255
        target[target == 255] = 0
        C = logit.shape[1]
        logit = logit.permute([0,2,3,1]).reshape([-1, C])[roi.reshape(-1), :]
        logit_correct = logit.gather(1, target[roi].reshape([-1, 1]).long())
        loss = torch.relu(logit).sum() - torch.relu(logit_correct).sum() + torch.relu(-logit_correct).sum()
        return loss / logit.shape[0]


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
