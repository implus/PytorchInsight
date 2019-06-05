## Installation

### Requirements

- Linux (tested on Ubuntu 16.04 and CentOS 7.2)
- Python 3.4+
- PyTorch 1.0
- Cython
- [mmcv](https://github.com/open-mmlab/mmcv) >= 0.2.2

### Install mmdetection

a. Install PyTorch 1.0 and torchvision following the [official instructions](https://pytorch.org/).

b. Compile cuda extensions.

```shell
cd mmdetection
pip install cython  # or "conda install cython" if you prefer conda
./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
```

c. Install mmdetection (other dependencies will be installed automatically).

```shell
python(3) setup.py install  # add --user if you want to install it locally
# or "pip install ."
```

Note: You need to run the last step each time you pull updates from github.
The git commit id will be written to the version number and also saved in trained models.

### Prepare COCO dataset.

It is recommended to symlink the dataset root to `$MMDETECTION/data`.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```
