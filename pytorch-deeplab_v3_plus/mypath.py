import os

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        data_root = os.environ['DATA_ROOT']
        if dataset == 'pascal':
            # folder that contains pascal/. It should have three subdirectories
            # called "JPEGImages", "SegmentationClassAug", and "pascal_2012_scribble"
            # containing RGB images, groundtruth, and scribbles respectively.
            return data_root + '/VOCdevkit/VOC2012/'
        elif dataset == 'sbd':
            return data_root + '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return data_root + '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return data_root + '/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
