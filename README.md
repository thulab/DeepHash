# DeepHash

DeepHash is a lightweight deep learning to hash library that implements state-of-the-art deep hashing/quantization algorithms. We will implement more representative deep hashing models continuously according to our released paper list. Specifically, we welcome other researchers to contribute deep hashing models into this toolkit based on our framework. We will announce the contribution in this project.

The implemented models include [Deep Quantization Network (DQN)](http://yue-cao.me/doc/deep-quantization-networks-dqn-aaai16.pdf), [Deep Hashing Network (DHN)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-hashing-network-aaai16.pdf), [Deep Visual-Semantic Quantization (DVSQ)](http://yue-cao.me/doc/deep-visual-semantic-quantization-cvpr17.pdf) and [Deep Cauchy Hashing (DCH)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-cauchy-hashing-cvpr18.pdf).

## Requirements

-  numpy==1.13.1
-  scipy==0.19.1
-  tensorflow==1.3.0
-  h5py==3.0.1
-  scikit-learn==0.19.0
-  python-opencv==3.0.1

<!--**TensorFlow Installation**-->

<!--Our hash learning lib requires Tensorflow (version 1.0+) to be installed.-->

<!--To install TensorFlow, simply run:-->
<!--```-->
<!--pip install tensorflow-->
<!--```-->
<!--or, with GPU-support:-->
<!--```-->
<!--pip install tensorflow-gpu-->
<!--```-->

<!--For more details see *[TensorFlow installation instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)*-->

<!--**Other Installation**-->

<!--To easily use our lib, we need to install scipy, python-opencv, h5py, scikit-learn, coverage.py and pytest by:-->

<!--```shell-->
<!--pip install scipy -->
<!--sudo apt-get install python-opencv-->
<!--pip install h5py-->
<!--pip install -U scikit-learn-->
<!--```-->

To use the algorithms implemented in `./DeepHash`, we need to add the path of `./DeepHash` to environment variables as:

```shell
export PYTHONPATH=/path/to/project/DeepHash/DeepHash:$PYTHONPATH
```             

## Data Preparation
In `data/cifar10/train.txt`, we give an example to show how to prepare image training data. In `data/cifar10/test.txt` and `data/cifar10/database.txt`, the list of testing and database images could be processed during predicting procedure. If you want to add other datasets as the input, you need to prepare `train.txt`, `test.txt` and `database.txt` as CIFAR-10 dataset.

## Get Started
The example of `$method` (DCH, DVSQ, DQN and DHN) can be run with the following command:
```shell
cd example/$method/
./train_val.sh
```

## DeepHash
* `./DeepHash/model/`: contains the implementation of models: dhn, dqn, dvsq and dch.
* `./DeepHash/architecture/`: contains the implementation of network structure, e.g. AlexNet.
* `./DeepHash/data_provider/`: contains the data reader implementation.
* `./DeepHash/evaluation/`: contains the implementation of evaluation criteria in search procedure, such as mAP, precision, recall and so on.
<!--**Data\_provider**-->
<!--**Architecture**-->
<!--**Model**-->
<!--**Evaluation**-->

## Methods
* `DCH`: Deep Cauchy Hashing for Hamming Space Retrieval, Yue Cao, Mingsheng Long, Bin Liu, Jianmin Wang, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018
* `DVSQ`: Deep Visual-Semantic Quantization for Efficient Image Retrieval, Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
* `DQN`: Deep Quantization Network for Efficient Image Retrieval, Yue Cao, Mingsheng Long, Jianmin Wang, Han Zhu, Qingfu Wen, AAAI Conference on Artificial Intelligence (AAAI), 2016
* `DHN`: Deep Hashing Network for Efficient Similarity Retrieval, Han Zhu, Mingsheng Long, Jianmin Wang, Yue Cao, AAAI Conference on Artificial Intelligence (AAAI), 2016

## Citations
If you find *DeepHash* is useful for your research, please consider citing the following papers:

    @InProceedings{cite:AAAI16DQN,
      Author = {Yue Cao and Mingsheng Long and Jianmin Wang and Han Zhu and Qingfu Wen},
      Publisher = {AAAI},
      Title = {Deep Quantization Network for Efficient Image Retrieval},
      Year = {2016}
    }
    
    @InProceedings{cite:AAAI16DHN,
      Author = {Han Zhu and Mingsheng Long and Jianmin Wang and Yue Cao},
      Publisher = {AAAI},
      Title = {Deep Hashing Network for Efficient Similarity Retrieval},
      Year = {2016}
    }
    
    @InProceedings{cite:CVPR17DVSQ,
      Title={Deep visual-semantic quantization for efficient image retrieval},
      Author={Cao, Yue and Long, Mingsheng and Wang, Jianmin and Liu, Shichen},
      Booktitle={CVPR},
      Year={2017}
    }
    
    @InProceedings{cite:CVPR18DCH,
      Title={Deep Cauchy Hashing for Hamming Space Retrieval},
      Author={Cao, Yue and Long, Mingsheng and Bin, Liu and Wang, Jianmin},
      Booktitle={CVPR},
      Year={2018}
    }


## Contacts
Maintainers of hash learning library:
* Yue Cao, Email: caoyue10@gmail.com
* Bin Liu, Email: liubinthss@gmail.com
