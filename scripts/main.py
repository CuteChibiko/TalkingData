import pandas as pd
import model_lib
from lib_util import get_target,get_opt
from read_data import read_data_ph1, read_csv

target=get_target()
print('start for',target)

train_df, test_df, numerical_patterns, cat_patterns = read_data_ph1()
predictors = numerical_patterns + cat_patterns
categorical = cat_patterns

is_val = (train_df['day'] == 9) & ((train_df['hour'] == 13) |(train_df['hour'] == 17) |(train_df['hour'] == 21))
val_df = train_df[is_val]
train_df = train_df[~is_val]

auc = model_lib.Predict(train_df,val_df,test_df,predictors,categorical,seed=get_opt('seed',2018))
print('validation auc:',auc)

test_df = test_df[['pred']].rename(columns={'pred': 'is_attributed'})
mapping = read_csv('../input/mapping.csv')
click_id = read_csv('../input/sample_submission.csv',usecols=['click_id'])
test_df = test_df.reset_index().merge(mapping, left_on='index', right_on='old_click_id', how='left')
test_df = click_id.merge(test_df,on='click_id',how='left')
outfile = '../csv/pred_test_'+target+'.csv'
print('writing to',outfile)
test_df[['click_id','is_attributed']].to_csv(outfile,index=False)
