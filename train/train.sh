#!/usr/bin/env bash

pkg=$HOME/.ros/handheld_object_log

if [[ ! -e $pkg ]]; then
    mkdir $pkg
fi

dir=snapshots
if [[ ! -e $dir ]]; then
    mkdir $dir
fi

export CAFFE_ROOT=/home/krishneel/caffe
export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
pkg_directory=$PWD/..
export PYTHONPATH=$pkg_directory/scripts/data_argumentation_layer:$PYTHONPATH
fname=$(date "+%Y-%m-%d-%H.%M-%S")

$CAFFE_ROOT/build/tools/caffe train --solver=solver.prototxt \
    --gpu=0 \
    --weights=/home/krishneel/caffe/models/ilsvrc14/vgg16/VGG_ILSVRC_16_layers.caffemodel \
    2>&1 | tee -a $pkg/handheld_object_$fname.log
