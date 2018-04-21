## Overall

The DeepHash package is a lightweight deep learning to hash library based on TensorFlow. This repository contains newly designed deep learning to hash and quantization methods, e.g. DHN, DQN, DVSQ and DCH. Besides, hash learning library provides high-level APIs and working examples for defining, training, fine-tuning and evaluating hashing models.

## Installation

**TensorFlow Installation**

Our hash learning lib requires Tensorflow (version 1.0+) to be installed.

To install TensorFlow, simply run:
```
pip install tensorflow
```
or, with GPU-support:
```
pip install tensorflow-gpu
```

For more details see *[TensorFlow installation instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)*

**Other Installation**

To easily use our lib, we need to install scipy, python-opencv, h5py, scikit-learn, coverage.py and pytest by:

```python
pip install scipy 
sudo apt-get install python-opencv
pip install h5py
pip install -U scikit-learn
pip install coverage
pip install pytest
```

To use the algorithms implemented in `./core`, we need to add the path of `./core` to environment variables as:

```python
export PYTHONPATH=/path/to/project/DeepHash/core:$PYTHONPATH
```             

## Data Preparation
In `data/cifar10/train.txt`, we give an example to show how to prepare image training data. In `data/cifar10/test.txt` and `data/cifar10/database.txt`, the list of testing and database images could be processed during predicting procedure. If you want to add other datasets as the input, you need to prepare `train.txt`, `test.txt` and `database.txt` as CIFAR-10 dataset.

## Get Started
The example of `$method` (DCH, DVSQ, DQN and DHN) can be run with the following command:
```
cd example/$method/
./train_val.sh
```

## Core
* `./core/model/`: contains the implementation of models: dhn, dqn, dvsq and dch.
* `./core/architecture/`: contains the implementation of network structure, e.g. AlexNet.
* `./core/data_provider/`: contains the data reader implementation.
* `./core/evaluation/`: contains the implementation of evaluation criteria in search procedure, such as mAP, precision, recall and so on.
<!--**Data\_provider**-->
<!--**Architecture**-->
<!--**Model**-->
<!--**Evaluation**-->

## Methods
* `DCH`: Deep Cauchy Hashing for Hamming Space Retrieval, Yue Cao, Mingsheng Long, Bin Liu, Jianmin Wang, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018
* `DVSQ`: Deep Visual-Semantic Quantization for Efficient Image Retrieval, Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
* `DQN`: Deep Quantization Network for Efficient Image Retrieval, Yue Cao, Mingsheng Long, Jianmin Wang, Han Zhu, Qingfu Wen, AAAI Conference on Artificial Intelligence (AAAI), 2016
* `DHN`: Deep Hashing Network for Efficient Similarity Retrieval, Han Zhu, Mingsheng Long, Jianmin Wang, Yue Cao, AAAI Conference on Artificial Intelligence (AAAI), 2016

## Contacts
Maintainers of hash learning library:
* Yue Cao, Email: caoyue10@gmail.com
* Bin Liu, Email: liubinthss@gmail.com
