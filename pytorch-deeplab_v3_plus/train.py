import numbers
import json
from tqdm import tqdm

import torch, torchvision
import torch.nn.functional as F

from modeling.deeplab import *
from dataloaders.utils import decode_seg_map_sequence, normalize_image_to_range
from dataloaders import make_data_loader
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


class TrainerBase(object):
    def __init__(self, args, nclass):
        self.args = args
        self.nclass = nclass

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Log program arguments
        self.writer.add_text("Args/experiment_dir", self.saver.experiment_dir)
        for key, value in vars(args).items():
            if isinstance(value, numbers.Number):
                self.writer.add_scalar("Args/" + key, value)
            else:
                self.writer.add_text("Args/" + key, str(value))
        self.writer.add_text("Args/All", json.dumps(vars(args), indent=4, sort_keys=True))

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs,
                                      # args.hidden_update or args.epochs,
                                      len(self.train_loader))


    def train_last_layer(self):
        print('\n=>Computing the last layer')
        self.model.eval()
        kwargs = {'num_workers': self.args.workers, 'pin_memory': True}
        train_loader = make_data_loader(self.args, None, **kwargs)[0]
        num_img_tr = len(train_loader)
        tbar = tqdm(train_loader)
        features = None
        count = None

        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image, target_cpu = sample['image'], sample['label']
                inside = target_cpu != 254
                croppings = inside.float()
                outside = target_cpu == 254
                target_cpu[outside] = 255
                image.transpose(0, 1)[:, outside] = 0
                target = target_cpu
                if self.args.cuda:
                    image, target = image.cuda(), target_cpu.cuda()
                target_long = target.long()

                output = self.model(image)
                last_layer = self.model.module.decoder.last_layer

                if features is None:
                    features = torch.zeros(
                        [last_layer.shape[1], self.nclass],
                        device=output.device
                    )
                    features2 = torch.zeros_like(features)
                    count = torch.zeros(
                        [1, self.nclass],
                        device=output.device
                    )


                for f, t in zip(last_layer, target_long):
                    f = F.interpolate(f.unsqueeze(0), size=image.size()[2:], mode='bilinear', align_corners=True).squeeze(0)
                    f2 = f.reshape((f.shape[0], -1))
                    t = t.reshape((-1,))
                    good = t < 255
                    f2 = f2[:, good]
                    t = t[good]
                    features.scatter_add_(1, t[None,:].repeat(f2.shape[0], 1), f2)
                    features2.scatter_add_(1, t[None,:].repeat(f2.shape[0], 1), f2 ** 2)
                    count += torch.bincount(t, minlength=self.nclass)[None,:]

                tbar.set_description('Computing last layer features, norm of sum: %f' % features.norm())

            features /= count
            # features2 -= (features2 - features ** 2 * count).sum(dim=1, keepdim=True) /  count.sum()
            # features2 = features2 / count - features ** 2
            features2 = (features2 - features ** 2 * count).sum(dim=1, keepdim=True) /  count.sum()
            print("Sigma shape:", features2.shape)
            print("Sigma range:", features2.min(), features2.max())
            print("Weight norm per class:", features.norm(dim=0) ** 2 / 2)
            print("Weight norm per feature:", features.norm(dim=1) ** 2 / 2)

            features2 = 0.5 * features2 ** -1
            for name, param in self.model.module.decoder.last_conv[-1].named_parameters():
                if name == 'weight':
                    param.data[...] = (features2 * features).transpose(0,1)[..., None, None]
                elif name == 'bias':
                    param.data[...] = -(features2 ** 0.5 * features).norm(dim=0) ** 2 / 2
                print(name, type(param), param.size())


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target[target==254]=255
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target.byte())
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu()
            target = target.cpu()
            pred = torch.argmax(pred, axis=1)

            if i < self.args.viz_images_per_epoch:
                vis_image = normalize_image_to_range(image.cpu())[0]
                vis_gt = decode_seg_map_sequence(target, dataset=self.args.dataset)[0]
                vis_pred = decode_seg_map_sequence(pred, dataset=self.args.dataset)[0]
                grid = torchvision.utils.make_grid([vis_image, vis_gt, vis_pred], 1)
                self.writer.add_image('val/Sample_%01d' % i, grid, epoch)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target.numpy(), pred.numpy())

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.val_loader.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            # self.saver.save_checkpoint({
            #     'epoch': epoch,
            #     'state_dict': self.model.module.state_dict(),
            #     'optimizer': self.optimizer.state_dict(),
            #     'best_pred': self.best_pred,
            # }, is_best)
