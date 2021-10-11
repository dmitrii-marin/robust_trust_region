from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        if split == 'train':
            self._cat_dir_full = os.path.join(self._base_dir, 'SegmentationClassAug')
            if 'full_supervision' in args and args.full_supervision:
                self._cat_dir = self._cat_dir_full
            else:
                # weak supervision with scribbles
                suffix = args.train_dataset_suffix
                if len(suffix) > 0:
                    print("Loading train masks with suffix '%s'" % suffix)
                self._cat_dir = os.path.join(self._base_dir, 'pascal_2012_scribble' + suffix)
        elif split == 'val':
            self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        #self._cat_dir = os.path.join(self._base_dir, 'pascal_2012_scribble_val_full')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        #_splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'SegmentationAug')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.categories_full = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image), _image
                assert os.path.isfile(_cat), _cat
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                if split == 'train':
                    _cat = os.path.join(self._cat_dir_full, line + ".png")
                    assert os.path.isfile(_cat), _cat
                    self.categories_full.append(_cat)


        assert (len(self.images) == len(self.categories))

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        mean_rgb = tuple((np.array(mean) * 255).astype(np.uint8))

        self.normalize = transforms.Compose([
                tr.ToTensor(),
                tr.Normalize(mean=mean, std=std),
            ])

        self.denormalize = tr.Denormalize(mean=mean, std=std)

        if 'no_aug' in self.args and self.args.no_aug:
          self.training_transform = transforms.Compose([
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                    image_fill=mean_rgb,
                    random=False,
                ),
                self.normalize,
            ])
        else:
          self.training_transform = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                    image_fill=mean_rgb,
                ),
                tr.RandomGaussianBlur(),
                self.normalize,
            ])

        self.val_transform = transforms.Compose([
                self.normalize,
            ])

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target, _target_full = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        if _target_full is not None:
            sample['label_full'] = _target_full

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        _target_full = Image.open(self.categories_full[index]) if len(self.categories_full) > 0 else None
        return _img, _target, _target_full

    def transform_tr(self, sample):
        return self.training_transform(sample)

    def transform_val(self, sample):
        return self.val_transform(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


class VOCProposalSegmentation(VOCSegmentation):
    def __init__(self, proposal_generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proposal_generator = proposal_generator

    def __getitem__(self, index):
        _img, _target, _target_full = self._make_img_gt_point_pair(index)
        _proposal, un, sm = self.proposal_generator(
            self.normalize({'image': _img})['image'],
            torch.tensor(np.array(_target)).byte(),
            index
        )
        # _proposal = Image.fromarray(_proposal[0,0].numpy(), 'L')
        sample = {'image': _img, 'label': _target,
            'label_proposal': _proposal, 'un': un, 'sm': sm}
        if _target_full is not None:
            sample['label_full'] = _target_full

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
