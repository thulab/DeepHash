#!/bin/bash

lr=$1
q_lambda=0.1
alpha=$2
dataset=nuswide_81 # coco # cifar10,  nuswide_81
log_dir=tflog

bias=0.0

gamma=$3

if [ -z "$4" ]; then
    gpu=0
else
    gpu=$4
fi

export TF_CPP_MIN_LOG_LEVEL=3
#                                                         lr  output  iter    q_lamb    alpha     dataset     gpu    log_dir bias   gamma
CUDA_VISIBLE_DEVICES=$gpu python train_val_script.py      $lr 64     2000    $q_lambda  $alpha   $dataset    0   $log_dir    $bias  $gamma
