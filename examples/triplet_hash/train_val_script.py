import numpy as np
import scipy.io as sio
import warnings
import data_provider.image as dataset
import model.triplet_hash.triplet_hash as model
import sys
from pprint import pprint
import os
import argparse

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Triplet Hashing')
parser.add_argument('--lr', '--learning_rate', default=0.00003, type=float)
parser.add_argument('--triplet_margin', default=30, type=float)
parser.add_argument('--select_strategy', default='margin', choices=['hard', 'all', 'margin', 'sign-margin', 'sign-hardneg', 'sign-prob'])
parser.add_argument('--output_dim', default=64, type=int)   # 256, 128
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--cq_lambda', default=0, type=float)
parser.add_argument('--q_lambda', default=0, type=float)
parser.add_argument('--n_subspace', default=4, type=int)
parser.add_argument('--n_subcenter', default=256, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--log_dir', default='tflog', type=str)
parser.add_argument('--dist_type', default='euclidean2', type=str)
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('-vb', '--val_batch_size', default=16, type=int)
parser.add_argument('--decay_step', default=10000, type=int)
parser.add_argument('--decay_factor', default=0.1, type=int)

tanh_parser = parser.add_mutually_exclusive_group(required=False)
tanh_parser.add_argument('--with_tanh', dest='with_tanh', action='store_true')
tanh_parser.add_argument('--without_tanh', dest='with_tanh', action='store_false')
parser.set_defaults(with_tanh=True)

parser.add_argument('--img_model', default='alexnet', type=str)
parser.add_argument('--model_weights', type=str,
                    default='../../core/architecture/single_model/pretrained_model/reference_pretrain.npy')
parser.add_argument('--finetune_all', default=True, type=bool)
parser.add_argument('--max_iter_update_b', default=3, type=int)
parser.add_argument('--max_iter_update_Cb', default=1, type=int)
parser.add_argument('--code_batch_size', default=500, type=int)
parser.add_argument('--n_part', default=20, type=int)
parser.add_argument('--triplet_thresold', default=64000, type=int)
parser.add_argument('--save_dir', default="./models/", type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--val_freq', default=1, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_21': 21, 'nuswide_81': 81, 'coco': 80, 'imagenet': 100}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000, 'nuswide_21': 5000, 'imagenet': 5000}
_dataset = args.dataset
args.R = Rs[_dataset]
args.label_dim = label_dims[_dataset]
args.img_tr = "/home/caoyue/data/{}/train.txt".format(_dataset)
args.img_te = "/home/caoyue/data/{}/test.txt".format(_dataset)
args.img_db = "/home/caoyue/data/{}/database.txt".format(_dataset)

pprint(vars(args))

query_img, database_img = dataset.import_validation(args.img_te, args.img_db)

if not args.evaluate:
    train_img = dataset.import_train(args.img_tr)
    model_weights = model.train(train_img, database_img, query_img, args)
    args.model_weights = model_weights
else:
    maps = model.validation(database_img, query_img, args)
    for key in maps:
        print(("{}\t{}".format(key, maps[key])))

pprint(vars(args))
