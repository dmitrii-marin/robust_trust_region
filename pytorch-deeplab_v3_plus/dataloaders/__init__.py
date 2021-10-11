from torch.utils.data import DataLoader, dataset
from dataloaders.datasets import combine_dbs, indexed_dataset
import numpy as np

def make_data_loader(args, proposal_generator=None, **kwargs):

    def wrap_dataset(set):
      if 'single_image_training' in args and args.single_image_training is not None:
        if args.single_image_training >= 0:
            indices = [args.single_image_training]
        else:
            state = np.random.RandomState(575393350)
            indices = state.choice(len(set), -args.single_image_training, replace=False)
        print("Training on subset of images %s" % (indices,))
        set = dataset.Subset(set, indices)
      return indexed_dataset.IndexedDataset(set)

    if 'train_shuffle' in args:
        shuffle = args.train_shuffle
    else:
        shuffle = False

    if args.dataset == 'pascal':
        from dataloaders.datasets import pascal
        if proposal_generator is None:
            train_set = pascal.VOCSegmentation(args, split='train')
        else:
            train_set = pascal.VOCProposalSegmentation(proposal_generator, args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_set = wrap_dataset(train_set)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = None

    elif args.dataset == 'cityscapes':
        from dataloaders.datasets import cityscapes
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(wrap_dataset(train_set), batch_size=args.batch_size, shuffle=shuffle, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'coco':
        from dataloaders.datasets import coco
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(wrap_dataset(train_set), batch_size=args.batch_size, shuffle=shuffle, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = None

    else:
        raise NotImplementedError

    return train_loader, val_loader, test_loader, num_class
