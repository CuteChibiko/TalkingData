#!/bin/sh -xe

path=../scripts

# build LGBM model
python -u $path/main.py model=LGBM_feat=lgbmBest_categoricalThreVal=10000_params=-,gbdt,0.39,0.05,305,7,auc,20,5,0,70,74,binary,0,0,5.0,1.0,200000,1,0
