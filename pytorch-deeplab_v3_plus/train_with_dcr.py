import os, sys
import argparse
import math
import time
from tqdm import tqdm

import numpy as np

import torchvision
import torch
import torch.nn.functional as F

from mypath import Path
from dataloaders import make_data_loader
from dataloaders.utils import decode_seg_map_sequence, normalize_image_to_range
from dataloaders.custom_transforms import denormalizeimage
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.metrics import Evaluator
from utils.proposal_generator import ProposalGeneratorFileCache
from utils.log_lin_softmax import log_lin_softmax
from train import TrainerBase

from DenseCRFLoss import DenseCRFLoss
import GridCRFLoss


def nll_error_loss(logits, seeds, error_labels, eps):
    N, C = logits.shape[:2]
    prob_log_mix = log_lin_softmax(eps / (C - 1), 1 - C * eps / (C - 1), logits, 1)
    if seeds is not None:
        prob_log_mix = prob_log_mix.permute(0,2,3,1)
        prob_log_mix[seeds != 255, :] = F.log_softmax(logits.permute(0,2,3,1)[seeds != 255, :], -1)
        prob_log_mix = prob_log_mix.permute(0,3,1,2)
    celoss = F.nll_loss(prob_log_mix, error_labels[:,0].long(), ignore_index=255)
    if seeds is not None:
        celoss *= (error_labels != 255).float().sum() / (seeds != 255).sum()
    celoss /= N
    return celoss


class Trainer(TrainerBase):
    def __init__(self, args):

        self.evaluator_full = None

        def ProposalGenerator(*args, **kwargs):
            return ProposalGeneratorFileCache(*args, **kwargs, eps=0)

        self.proposal_generator = None
        if args.use_dcr:
            if args.proposals is not None:
                self.proposal_generator = ProposalGenerator(None, path=args.proposals)
                print("No explicit proposal generator")
            else:
                if args.use_dcr == "AlphaExpansion":
                    import AlphaExpansion
                    generator = AlphaExpansion.AlphaExpansion(
                        max_iter=args.gc_max_iters,
                        potts_weight=args.potts_weight,
                        ce_weight=args.tr_weight,
                        restrict=args.tr_restricted,
                        scale=args.gc_scale
                    )
                    if args.alpha_use_edge_predictor:
                        from PIL import Image
                        old_generator = generator
                        path = args.alpha_use_edge_predictor
                        def _decorator(unary, image, *args, **kwargs):
                            img = Image.open(path + "/%05d.png" % kwargs['index'])
                            img = np.array(img, np.float32)
                            edges = torch.tensor(img) / 255.
                            return old_generator(unary, edges[None, None], *args, **kwargs)
                        generator = _decorator
                self.proposal_generator = ProposalGenerator(generator)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, nclass \
            = make_data_loader(args, self.proposal_generator, **kwargs)

        super().__init__(args, nclass)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        v=args.v)
        self.freeze_bn = args.freeze_bn

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * args.last_layer_mult_lr}]

        #error model
        if self.args.tr_error_model == 'Const':
            self.error_prob = lambda: torch.tensor(self.args.tr_error_prob)
        elif self.args.tr_error_model == 'Uniform':
            x = -math.log(1/self.args.tr_error_prob - 1)
            log_error_prob = torch.tensor(x, requires_grad=True)
            train_params.append({'params': [log_error_prob], 'lr': args.lr})
            self.error_prob = lambda: torch.sigmoid(log_error_prob)
        elif self.args.tr_error_model == 'Poly0':
            start = 1 - 1.0 / self.nclass
            target = -math.log(1/self.args.tr_error_prob - 1)
            power = 1
            self.error_prob = lambda: \
                start + (end - start) * (self.scheduler.T / self.scheduler.N) ** power

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.tr_extra_criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='ce')
        self.model, self.optimizer = model, optimizer

        relaxation = {
            'bilinear': GridCRFLoss.BilinearPottsRelaxation,
            'squared': GridCRFLoss.SquaredPottsRelaxation,
            'tv': GridCRFLoss.TVPottsRelaxation,
        }[args.relaxation]

        self.gridcrf = GridCRFLoss.GridPottsLoss(weight=args.potts_weight,
            scale_factor=args.rloss_scale, relaxation=relaxation)
        self.pce = nn.CrossEntropyLoss(ignore_index=255)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        if args.precompute_last_layer:
            self.train_last_layer()


    def training(self, epoch):
        train_loss = 0.0
        train_celoss = 0.0
        train_crfloss = 0.0

        train_smooth = 0.0
        train_unary = 0.0
        train_relaxed_un = 0.0
        train_relaxed_sm = 0.0

        self.evaluator.reset()
        self.evaluator_full = None

        if self.args.use_dcr:
            train_smooth_p1 = 0.0
            train_unary_p1 = 0.0
            train_smooth_p1_upsample = 0.0
            train_unary_p1_upsample = 0.0
            if self.args.proposals:
                self.proposal_generator.update_model(
                    self.model.module, True)
            else:
                if self.args.use_dcr == "AlphaExpansion":
                    self.proposal_generator.update_model(
                        self.model.module, False if epoch > 0 else None)
                    self.proposal_generator.alpha_expansion.max_iter = \
                            5 if epoch % self.args.hidden_update == 0 else 0
                else:
                    self.proposal_generator.update_model(
                        self.model.module,
                        True if epoch % self.args.hidden_update != 0 else None
                    )

        self.model.train()
        if self.freeze_bn:
            freeze_batchnorm(self.model)

        print('\n=>Epoches %i, learning rate = %.4f, previous best = %.4f'
            % (epoch, self.scheduler.actual_lr, self.best_pred))

        num_img_tr = len(self.train_loader)
        softmax = nn.Softmax(dim=1)
        self.evaluator.reset()

        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            iter = i + num_img_tr * epoch
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

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            probs = softmax(output)

            if self.args.use_dcr:
                hidden = sample['label_proposal']
                un, sm = sample['un'], sample['sm']
                if self.args.cuda:
                    hidden = hidden.cuda()

                hidden[hidden == 254] = 255

                if self.args.tr_soften:
                    if hidden.ndim != 4:
                        hidden.unsqueeze_(1)
                    if hidden.shape[1] == 1:
                        bad = hidden == 255
                        hidden[bad] = 0
                        hidden = torch.zeros_like(output).scatter_(1, hidden.long(), 1)
                        hidden_perm = hidden.permute([0,2,3,1])
                        hidden_perm_shape = hidden_perm.shape
                        hidden_perm = hidden_perm.reshape(-1, hidden.shape[1])
                        hidden_perm[bad.reshape(-1)] = 255
                        hidden = hidden_perm.reshape(hidden_perm_shape).permute([0,3,1,2])
                        del hidden_perm

                    hidden[(hidden != 255) & (target[:,None] == 255)] *= 1 - self.args.tr_soften
                    hidden[(hidden != 255) & (target[:,None] == 255)] += self.args.tr_soften / output.shape[1]

                if self.args.tr_error_model in ['Const', 'Uniform']:
                    eps = self.error_prob()
                    self.writer.add_scalar('train_iter/error_prob', eps.item(), iter)
                    celoss = nll_error_loss(output, target, hidden, eps)
                else:
                    celoss = self.criterion(output, hidden)

                if self.args.use_pce_at_tr > 0:
                    celoss += self.tr_extra_criterion(output, target) * self.args.use_pce_at_tr

                loss = celoss + 0
            else:
                if self.args.relaxation_target == "Prob":
                    gridcrf_target = probs
                elif self.args.relaxation_target == "LogProb":
                    gridcrf_target = F.log_softmax(output, dim=1)
                elif self.args.relaxation_target == "Logits":
                    gridcrf_target = output
                else:
                    raise KeyError
                init_rel_sm = self.gridcrf(image, gridcrf_target, croppings.cuda())

                if self.args.tr_error_model in ['Const', 'Uniform']:
                    eps = self.error_prob()
                    self.writer.add_scalar('train_iter/error_prob', eps.item(), iter)
                    init_rel_un = nll_error_loss(output, None, target[:,None], eps)
                else:
                    init_rel_un = self.criterion(output, target_long)
                loss = init_rel_sm + init_rel_un
            train_loss += loss.item()

            self.writer.add_scalar('train_iter/total_gap_loss', loss.item(), iter)
            loss.backward()
            self.optimizer.step()

            if 'label_full' in sample:
                self.evaluator_full = self.evaluator_full or Evaluator(self.nclass)
                self.evaluator_full.add_batch(sample['label_full'].numpy(), torch.argmax(output, 1).cpu().numpy())
            self.evaluator.add_batch(target_cpu.numpy(), torch.argmax(output, 1).cpu().numpy())
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            self.writer.add_scalar('train_iter/learning_rate', self.scheduler.actual_lr, iter)
            self.writer.add_scalar('train_iter/loss', loss.item(), iter)

            if self.args.v == '3.2' and i % max(1, num_img_tr // 5) == 0:
                self.writer.add_histogram("train_iter/LastConvFeatNorm", list(self.model.module.decoder.last_conv.parameters())[0].norm(dim=0) , i)

            # Show 5 * 9 inference results each epoch
            if self.args.viz_images_per_epoch and i % max(1, num_img_tr // self.args.viz_images_per_epoch) == 0:
                global_step = i + num_img_tr * epoch
                prefix = "e%02d/" % epoch
                if self.args.use_dcr:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, hidden, output, i, prefix=prefix)
                else:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i, prefix=prefix)
                grid = torchvision.utils.make_grid(
                    decode_seg_map_sequence(target[:9].detach().cpu().numpy(), dataset=self.args.dataset),
                    3, normalize=False, range=(0, 255)
                )
                self.writer.add_image(prefix + "Seeds", grid, i)
                self.writer.add_histogram(prefix + "PredHist", F.log_softmax(output, dim=1), i)
                best_class = torch.argmax(probs[:9].detach(), dim=1)
                best_prob = torch.max(probs[:9].detach(), dim=1, keepdim=True)[0]

                grid = torchvision.utils.make_grid(
                    decode_seg_map_sequence(best_class.cpu().numpy(), dataset=self.args.dataset) * best_prob.cpu(),
                    3, normalize=False, range=(0, 255)
                )
                self.writer.add_image(prefix + "PredictionCertanty", grid, i)
        self.writer.add_scalar('train/mIoU', self.evaluator.Mean_Intersection_over_Union(), epoch)
        if self.evaluator_full:
            self.writer.add_scalar('train/mIoU_full', self.evaluator_full.Mean_Intersection_over_Union(), epoch)

        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('train_gd/unary_loss', train_relaxed_un, epoch)
        self.writer.add_scalar('train_gd/smooth_loss', train_relaxed_sm, epoch)
        self.writer.add_scalar('train_gd/total_loss', train_relaxed_un + train_relaxed_sm, epoch)

        if self.args.use_dcr:
            self.writer.add_scalar('train_p1/unary_loss', train_unary_p1, epoch)
            self.writer.add_scalar('train_p1/smooth_loss', train_smooth_p1, epoch)
            self.writer.add_scalar('train_p1/total_loss', train_unary_p1 + train_smooth_p1, epoch)
            self.writer.add_scalar('train_p1_up/unary_loss', train_unary_p1_upsample, epoch)
            self.writer.add_scalar('train_p1_up/smooth_loss', train_smooth_p1_upsample, epoch)
            self.writer.add_scalar('train_p1_up/total_loss', train_unary_p1_upsample + train_smooth_p1_upsample, epoch)

        print('[Epoch: %d, numImages: %5d]' % (epoch + 1, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        sys.stdout.flush()

        #if self.args.no_val:
        if self.args.save_interval:
            # save checkpoint every interval epoch
            is_best = False
            if (epoch + 1) % self.args.save_interval == 0:
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best, filename='checkpoint_epoch_{}.pth.tar'.format(str(epoch+1)))


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--train_dataset_suffix', type=str, default='',
                        help='train mask directory suffix')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='l2',
                        choices=['ce', 'focal', 'l2', 'l1', 'margin0'],
                        help='loss func type (default: l2)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--last-layer-mult-lr', type=float, default=10,
                        help='last layer learning rate multiplier')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # model saving option
    parser.add_argument('--save-interval', type=int, default=None,
                        help='save model interval in epochs')
    parser.add_argument('--viz-images-per-epoch', type=int, default=5,
                        help='Number of viz images to save per epoch')

    # rloss options
    parser.add_argument('--densecrfloss', type=float, default=0,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale',type=float,default=1.0,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb',type=float,default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy',type=float,default=80.0,
                        help='DenseCRF sigma_xy')
    parser.add_argument('--relaxation', type=str, default='bilinear',
                        choices=['bilinear', 'squared', 'tv'],
                        help='Potts relaxation type (default: bilinear)')
    parser.add_argument('--relaxation-target', type=str, default='Prob',
                        choices=['Prob', 'Logits', 'LogProb'])
    parser.add_argument('--full-supervision', action='store_true', default=False)

    # dcr settings
    parser.add_argument('--use-dcr', type=str, default=None,
                        choices=[None, 'AlphaExpansion'],
                        help='Type of DCR/Trust-Region to use')
    parser.add_argument('--alpha-use-edge-predictor',type=str,default=None)
    parser.add_argument('--proposals',type=str,default=None)
    parser.add_argument('--tr-soften',type=float,default=0.0)
    parser.add_argument('--tr-error-model',type=str,default=None,
                        choices=['Const', 'Uniform', 'Poly0', 'ADM'])
    parser.add_argument('--tr-error-prob',type=float,default=0.5)
    parser.add_argument('--gc-max-iters',type=int,default=5,
                        help='Maximum number of graph cut iterations')
    parser.add_argument('--gc-scale',type=float,default=1,
                        help='Scale input to graph cut')
    parser.add_argument('--potts-weight',type=float,default=1.0,
                        help='Weight of potts term')
    parser.add_argument('--tr-weight',type=float,default=1.0,
                        help='Weight of TR term')
    parser.add_argument('--tr-restricted', action='store_true', default=False)
    parser.add_argument('--hidden-update',type=int,default=None,
                        help='Epoch frequency of phase1 solution updates')

    parser.add_argument('--use-pce-at-tr', type=float, default=0,
                        help='whether to use SBD dataset (default: 0)')
    parser.add_argument('--single-image-training', type=int, default=None)
    parser.add_argument('--train-shuffle', type=int, default=1)
    parser.add_argument('--no-aug', action='store_true', default=False)
    parser.add_argument('--use-linear-relaxation', action='store_true', default=False)

    parser.add_argument('--entropy-loss', type=float, default=0.0)

    parser.add_argument('--precompute-last-layer', action='store_true', default=False)
    parser.add_argument('--v', type=str, default=None)

    args = parser.parse_args()

    args.train_shuffle = bool(args.train_shuffle)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    if not trainer.args.no_val:
        trainer.validation(0)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        start_time = time.time()
        trainer.training(epoch)
        trainer.writer.add_scalar('train/time_per_epoch', time.time() - start_time, epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch + 1)

    trainer.writer.close()

if __name__ == "__main__":
   main()
