# [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)



## models and scores
model definition can be found in [scripts/model_lib.py](https://github.com/CuteChibiko/TalkingData/blob/master/scripts/model_lib.py)

  - **model1** LGBM with 83 (76 numerical, 7 categorical) features.

  - **model2** keras with 27(18 numerical, 9 categorical) features, You can see network structure in [model.png](https://github.com/CuteChibiko/TalkingData/blob/master/model.png)


|model|private score|public score|
|---|---|---|
|model1  |0.9836325|0.9828896|
|model2  |0.9830595|0.9822785|

## feature engineering and scripts
Most of these features have already been discussed on the [kaggle forum](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion).


  - counting features
    - mk_feat_count.py
    - mk_feat_count_time.py
    - mk_feat_countRatio.py

  - cumulative count
    - mk_feat_cumcount.py
    - mk_feat_recumcount.py
    - mk_feat_cumratio.py

  - time to next click
    - mk_feat_nextClick_leak_day.py
    - mk_feat_nextClick_filter.py

  - time bucket count.(make multiple time intervals, and count the number of buckets which the IP exists)
    - mk_feat_rangecount.py
    - mk_feat_rangecount_minute.py

  - variance
    - mk_feat_var.py

  - common IP
    - mk_feat_common_ip.py

  - unique count
    - mk_feat_uniq_count2.py

  - target encoding: woe
    - mk_feat_woe_all_prev.py
    - mk_feat_woe_bound.py
    
Features will be calculated once and saved to disk.

Importance from LGBM is found in [importance.txt](https://github.com/CuteChibiko/TalkingData/blob/master/importance.txt).


## Requirements
I used following environment

Hardware:
  - Memory: 256GB RAM, 256GB SWAP
  - CPU: 20 core, 2.10GHz
  - GPU: 1080Ti
  
Python3 packages:
  - numpy==1.14.2
  - pandas==0.22.0
  - lightgbm==2.1.0
  - keras==2.1.5

## How to run

At first, put sample_submission.csv test.csv test_supplement.csv train.csv to input directory.

Then run run.sh as follows,

`$ cd scripts/`

` $ ./run_mk_feats.sh`

` $ ./run_mk_model1.sh`

` $ ./run_mk_model2.sh`

 Output prediction files will be in csv directory.
 
 It took about one day for feature extraction(run_mk_feats.sh).
 
 It needs large memory(~256GB) to build model1(run_mk_model1.sh), sorry.
 
 GPU is required to build model2(run_mk_model2.sh)
 

 
 
