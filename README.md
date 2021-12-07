# Robust Trust Region for Weakly Supervised Segmentation

This is an implementation for the paper:
* [Dmitrii Marin](https://dmitriimarin.info) and [Yuri Boykov](https://cs.uwaterloo.ca/~yboykov/). **Robust Trust Region for Weakly Supervised Segmentation**. _In International Conference on Computer Vision (ICCV), 2021_

[[paper page]](https://openaccess.thecvf.com/content/ICCV2021/html/Marin_Robust_Trust_Region_for_Weakly_Supervised_Segmentation_ICCV_2021_paper.html) [[arxiv]](https://arxiv.org/abs/2104.01948) [[poster]](https://drive.google.com/file/d/1bhO0XoDJviqdRC1mnIxWuHD_li06Krgy/view?usp=sharing) [[video]](https://drive.google.com/file/d/1MLd3c-fpm2K3hgYyWYFFxW3Ve8FznfD2/view?usp=sharing)

If you find this code useful in your research, consider citing (bibtex):
```bibtex
@InProceedings{Marin_2021_ICCV,
    author    = {Marin, Dmitrii and Boykov, Yuri},
    title     = {Robust Trust Region for Weakly Supervised Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6608-6618}
}
```

|<img src="https://user-images.githubusercontent.com/3115577/136873099-8b708f5f-592c-4c50-b729-f9e52462fd48.png" style="width: 70%;" alt="Results on ScribbleSup dataset. We are the best!">|
|:--:|
|Results on ScribbleSup dataset using Deeplab-V3+ and MobileNet-V2. Our Robust Trust Region (Grid-TR) outperforms other weakly-supervised methods (gradient descent - Grid-DG) and objectives ([Dense-GD](https://github.com/meng-tang/rloss/), PCE-GD)|


## Prerequisites

### Environment

The code uses the following python packages: pytorch, tqdm, scipy, tensorboardX.

### ScribbleSup dataset

Download original PASCAL VOC 2012 dataset:
http://host.robots.ox.ac.uk/pascal/VOC/.

Download Scribble annotations: http://cs.uwaterloo.ca/~m62tang/rloss/pascal_2012_scribble.zip, the original dataset can be found [here](https://jifengdai.org/downloads/scribble_sup/).

### Building and installing python extensions

[Alpha-expansion code (GCO)](https://github.com/dmitrii-marin/alpha_expansion) for GridCRF regularized loss with Robust Trust Region:
```shell
cd wrapper/alpha_expansion
swig -python -c++ alpha_expansion.i
python setup.py install
```

Bilateral filtering for the [DenseCRF regularized loss](https://github.com/meng-tang/rloss/):
```shell
cd wrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
```

### Loading pre-trained models

All TR models require pretraining with partial cross-entropy. The models can be downloaded using the links:

| Level of supervision | % of image pixels labeled |Description | Link |
|--|--|--|--|
| full supervision | ~100% | mobilenet-v2 + deeplab-v3+ | [45MB](https://cs.uwaterloo.ca/~d2marin/models/highorder_backprop_iccv21/mn_checkpoint_full_epoch_60.pth.tar)|
| original (full-length scribbles) | ~3% | mobilenet-v2 + deeplab-v3+ | [45MB](https://cs.uwaterloo.ca/~d2marin/models/highorder_backprop_iccv21/mn_checkpoint_epoch_60.pth.tar) |
| 80%-length scribbles             | | mobilenet-v2 + deeplab-v3+ | [45MB](https://cs.uwaterloo.ca/~d2marin/models/highorder_backprop_iccv21/mn_checkpoint0.8_epoch_60.pth.tar) |
| 50%-length scribbles             | | mobilenet-v2 + deeplab-v3+ | [45MB](https://cs.uwaterloo.ca/~d2marin/models/highorder_backprop_iccv21/mn_checkpoint0.5_epoch_60.pth.tar) |
| 30%-length scribbles             | ~1% | mobilenet-v2 + deeplab-v3+ | [45MB](https://cs.uwaterloo.ca/~d2marin/models/highorder_backprop_iccv21/mn_checkpoint0.3_epoch_60.pth.tar) |
| clicks (0%-length scribbles)     | | mobilenet-v2 + deeplab-v3+ | [45MB](https://cs.uwaterloo.ca/~d2marin/models/highorder_backprop_iccv21/mn_checkpoint0_epoch_60.pth.tar) |

## Training with Robust Trust Region

Run the following script:
```shell
bash full_tr_error_const.sh 
```

The script is controlled by the following environmental variables:

* ```suffix``` — one of _empty string, 0.8, 0.5, 0.3, 0_. Determines the level of supervision, see pre-trained models description.
* ```ERROR_PROB``` — the robustness parameter ε. Must be between 0 and 1. See paper for details.
* ```LR``` — the base learning rate (default ```0.0007```).
* ```TR_WEIGHT``` — the weight of the unary term in alpha-expansion optimization. See stage-A in the paper.
* ```HIDDEN_UPDATE``` — the number of iterations of alpha expansion (defualt ```5```).

### Parameter configuration for different levels of supervision
| Level of supervision | ```suffix``` | ```ERROR_PROB``` | ```TR_WEIGHT```
|--|--|--|--|
| original (full-length scribbles)  | \<empty string\>| 0.94400 | 0.05  |
| 80%-length scribbles              | 0.8           | 0.94600 | 0.075 |
| 50%-length scribbles              | 0.5           | 0.95140 | 0.1   |
| 30%-length scribbles **(default)**| 0.3           | 0.95160 | 0.1   |
| clicks (0%-length scribbles)      | 0             | 0.95220 | 0.1   |

For example, the following will train a model using fulllength scribbles:
```shell
suffix= ERROR_PROB=0.94400 TR_WEIGHT=0.05 bash full_tr_error_const.sh
```
