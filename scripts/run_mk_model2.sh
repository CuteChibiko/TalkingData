#!/bin/sh -xe

path=../scripts

# build LGBM model

python $path/main.py BatchNormalization=on_sameNDenseAsEmb=off_model=keras_feat=kerasBest_params=-,20000,1000,1,0.2,100,2,0.001,0.0001,0.001,100,2,3
