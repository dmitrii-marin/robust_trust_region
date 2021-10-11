import torch
import torch.nn.functional as F
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = torch.tensor(mean)[:,None,None]
        self.std = torch.tensor(std)[:,None,None]

    def __call__(self, sample):
        result = {}
        for im_key in sample:
            if 'image' in im_key:
                img = sample[im_key]
                img /= 255.0
                img -= self.mean
                img /= self.std
                result[im_key] = img
            else:
                result[im_key] = sample[im_key]

        return result


class Denormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = torch.tensor(mean)[:,None,None] * 255
        self.std = torch.tensor(std)[:,None,None] * 255

    def __call__(self, sample):
        result = {}
        for im_key in sample:
            if 'image' in im_key:
                img = sample[im_key] * self.std + self.mean
                torch.clamp(img, 0, 255, out=img)
                result[im_key] = img
            else:
                result[im_key] = sample[im_key]

        return result


class NormalizeImage(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        result = {}
        for key in sample:
            val = sample[key]
            if not torch.is_tensor(val):
                val = np.array(val).astype(np.float32)
                if 'image' in key:
                    val = val.transpose((2, 0, 1))
                val = torch.from_numpy(val).float()
            result[key] = val

        return result


class ToPIL(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        result = {}
        for key in sample:
            val = sample[key]
            val = val.numpy().astype(np.float32)
            if 'image' in key:
                val = val.transpose((2, 0, 1))
            val = torch.from_numpy(val).float()
            result[key] = val

        return result


class ToTensorImage(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            return sample
        return { key: self.flip(sample[key]) if 'image' in key or 'label' in key else sample[key]
                for key in sample }

    def flip(self, datum):
        if torch.is_tensor(datum):
            return datum.flip(-1)
        return datum.transpose(Image.FLIP_LEFT_RIGHT)


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        return { key: sample[key].rotate(rotate_degree, Image.BILINEAR) if 'image' in key or 'label' in key else sample[key]
                for key in sample }


class RandomGaussianBlur(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            return sample

        result = {}
        for key in sample:
            val = sample[key]
            if key == 'image':
                val = val.filter(ImageFilter.GaussianBlur(
                    radius=random.random()))
            result[key] = val
        return result


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, label_fill=254, image_fill=0, random=True):
        self.base_size = base_size
        self.crop_size = crop_size
        self.label_fill = label_fill
        self.image_fill = image_fill
        self.random = random

    def __call__(self, sample):
        # random scale (short edge)
        w, h = sample['image'].size
        short_size = min(w, h)
        if self.random:
            short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        for key in sample:
            if 'image' in key:
                sample[key] = sample[key].resize((ow, oh), Image.BILINEAR)
            elif 'label' in key:
                if torch.is_tensor(sample[key]):
                    sample[key] = F.interpolate(sample[key][None,...], size=(oh, ow))[0]
                else:
                    sample[key] = sample[key].resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            for key in sample:
                fill = None
                if 'image' in key:
                    fill = self.image_fill
                elif 'label' in key:
                    fill = self.label_fill
                if fill is not None:
                    padw2, padh2 = padw // 2, padh // 2
                    if torch.is_tensor(sample[key]):
                        sample[key] = F.pad(
                            sample[key],
                            (padw2, padw - padw2, padh2, padh - padh2),
                            value=fill,
                        )
                    else:
                        sample[key] = ImageOps.expand(
                            sample[key],
                            border=(padw2, padh2, padw - padw2, padh - padh2),
                            fill=fill,
                        )
        # random crop crop_size
        w, h = sample['image'].size
        x1 = random.randint(0, w - self.crop_size) if self.random else (w - self.crop_size) // 2
        y1 = random.randint(0, h - self.crop_size) if self.random else (h - self.crop_size) // 2
        for key in sample:
            fill = None
            if 'image' in key or 'label' in key:
                if torch.is_tensor(sample[key]):
                    sample[key] = sample[key][..., y1:y1 + self.crop_size, x1:x1 + self.crop_size]
                else:
                    sample[key] = sample[key].crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return sample


class Pad(object):
    def __init__(self, crop_size, im_fill=0, lb_fill=254):
        self.crop_size = crop_size
        self.lb_fill = lb_fill
        self.im_fill = im_fill

    def __call__(self, sample):
        result = {}
        for key in sample:
            if 'label' not in key and 'image' not in key:
                result[key] = sample[key]
                continue
            arr = sample[key]
            oh, ow = arr.shape[-2:]
            fill = self.lb_fill if 'label' in key else self.im_fill
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            arr = F.pad(
                arr,
                (padw//2, padw - padw//2, padh//2, padh - padh//2),
                value=fill,
            )
            result[key] = arr
        return result


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCropImage(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        result = {}
        for key in sample:
            if 'label' not in key and 'image' not in key:
                result[key] = sample[key]
                continue
            arr = sample[key]
            method = Image.NEAREST if 'label' in key else Image.BILINEAR
            arr = arr.resize(self.size, method)
            result[key] = arr
        return result


def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *= 255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)
