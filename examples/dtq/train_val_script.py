import numpy as np
import scipy.io as sio
import warnings
import data_provider.image as dataset
import model.dtq as model
from pprint import pprint
import os
import argparse

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Triplet Hashing')
parser.add_argument('--lr', '--learning-rate', default=0.00003, type=float)
parser.add_argument('--triplet-margin', default=30, type=float)
parser.add_argument('--select-strategy', default='margin', choices=['hard', 'all', 'margin'])
parser.add_argument('--output-dim', default=64, type=int)   # 256, 128
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--cq-lambda', default=0, type=float)
parser.add_argument('--subspace', default=4, type=int)
parser.add_argument('--subcenter', default=256, type=int)
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'coco', 'nuswide_81'])
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--log-dir', default='tflog', type=str)
parser.add_argument('--dist-type', default='euclidean2', type=str,
                    choices=['euclidean2', 'cosine', 'inner_product', 'euclidean'])
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-vb', '--val-batch-size', default=16, type=int)
parser.add_argument('--decay-step', default=10000, type=int)
parser.add_argument('--decay-factor', default=0.1, type=int)

tanh_parser = parser.add_mutually_exclusive_group(required=False)
tanh_parser.add_argument('--with-tanh', dest='with_tanh', action='store_true')
tanh_parser.add_argument('--without-tanh', dest='with_tanh', action='store_false')
parser.set_defaults(with_tanh=True)

parser.add_argument('--img-model', default='alexnet', type=str)
parser.add_argument('--model-weights', type=str,
                    default='../../DeepHash/architecture/pretrained_model/reference_pretrain.npy')
parser.add_argument('--finetune-all', default=True, type=bool)
parser.add_argument('--max-iter-update-b', default=3, type=int)
parser.add_argument('--max-iter-update-Cb', default=1, type=int)
parser.add_argument('--code-batch-size', default=500, type=int)
parser.add_argument('--n-part', default=20, type=int)
parser.add_argument('--triplet-thresold', default=64000, type=int)
parser.add_argument('--save-dir', default="./models/", type=str)
parser.add_argument('--data-dir', default="~/data/", type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--val-freq', default=1, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

label_dims = {'cifar10': 10, 'nuswide_81': 81, 'coco': 80}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000}
args.R = Rs[args.dataset]
args.label_dim = label_dims[args.dataset]

args.img_tr = os.path.join(args.data_dir, args.dataset, "train.txt")
args.img_te = os.path.join(args.data_dir, args.dataset, "test.txt")
args.img_db = os.path.join(args.data_dir, args.dataset, "database.txt")

pprint(vars(args))

data_root = os.path.join(args.data_dir, args.dataset)
query_img, database_img = dataset.import_validation(data_root, args.img_te, args.img_db)

if not args.evaluate:
    train_img = dataset.import_train(data_root, args.img_tr)
    model_weights = model.train(train_img, database_img, query_img, args)
    args.model_weights = model_weights
else:
    maps = model.validation(database_img, query_img, args)
    for key in maps:
        print(("{}\t{}".format(key, maps[key])))

pprint(vars(args))
