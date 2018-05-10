import time
import os
import numpy as np
import pandas as pd
import pytz

nrows=100000
nrows=None

input_dir = '../input'
work_dir  = '../work'
dtype_train = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'is_attributed' : 'uint8',
        'click_time': object,
        }
dtype_test = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'click_id'      : 'uint32',
        'click_time': object,
        }

train_df = pd.read_csv(input_dir+"/train.csv", dtype=dtype_train, usecols=dtype_train.keys(), nrows=nrows, parse_dates=['click_time'])
test_df = pd.read_csv(input_dir+"/test_supplement.csv", dtype=dtype_test, usecols=dtype_test.keys(), nrows=nrows, parse_dates=['click_time'])
train_df['click_id'] = 0

# need to check
ans_file = "../input/ans_full.csv"
if os.path.exists(ans_file):
    df_ans_full = pd.read_csv(ans_file,nrows=nrows)
    test_df['is_attributed'] = df_ans_full['is_attributed']
else:
    test_df['is_attributed'] = 0
len_train = len(train_df)
df = train_df.append(test_df).reset_index()
cst = pytz.timezone('Asia/Shanghai')
df['datetime'] = pd.to_datetime(df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
df['day'] = df.datetime.dt.day.astype('uint8')
df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
df = df.sort_values(['click_time','is_attributed','click_id'])[['click_time','day','ip','app','device','os']]


name = 'nextClickLeakDay'
df[name] = (df.groupby(['day', 'ip', 'app', 'device', 'os']).click_time.shift(-1) - df.click_time+1).fillna(999999).astype(int)
out_df = df[[name]].sort_index()
out_df[len_train:].to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
out_df[:len_train].to_csv(work_dir + '/train_' + name + '.csv', index=False)
print(work_dir + '/test_supplement_' + name + '.csv')
print(work_dir + '/train_' + name + '.csv')

name = 'nextNextClickLeakDay'
df[name] = (df.groupby(['day', 'ip', 'app', 'device', 'os']).click_time.shift(-2) - df.click_time+1).fillna(999999).astype(int)
out_df = df[[name]].sort_index()
out_df[len_train:].to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
out_df[:len_train].to_csv(work_dir + '/train_' + name + '.csv', index=False)
print(work_dir + '/test_supplement_' + name + '.csv')
print(work_dir + '/train_' + name + '.csv')
