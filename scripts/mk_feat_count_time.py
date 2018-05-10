import pandas as pd
import sys
import pytz
import pickle
import os
import gc

nrows=10000
nrows=None

input_dir = '../input'
work_dir  = '../work'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
train_df = pd.read_csv(input_dir+"/train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], nrows=nrows)
test_df = pd.read_csv(input_dir+"/test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'], nrows=nrows)
test_org_df = pd.read_csv(input_dir+"/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'], nrows=nrows)

len_train = len(train_df)
df=train_df.append(test_df)
df.fillna(0,inplace=True)

#######################
def add_col(df,ptn):
    print('start for',ptn)
    name = "count_" + ptn
    dummy = 'is_attributed'
    cols = ptn.split("_")
    cols_with_dummy = cols.copy()
    cols_with_dummy.append(dummy)
    gp = df[cols_with_dummy].groupby(by=cols)[[dummy]].count().reset_index().rename(index=str, columns={dummy: name})
    _df = df.merge(gp, on=cols, how='left')
    _df[[name]][len_train:].to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
    _df[[name]][:len_train].to_csv(work_dir + '/train_' + name + '.csv', index=False)
    print('########### done for: ' + name + ' ###########')
    print(work_dir + '/test_supplement_' + name + '.csv')
    print(work_dir + '/train_' + name + '.csv')

cst = pytz.timezone('Asia/Shanghai')
df['click_time'] = pd.to_datetime(df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
df['hour'] = df.click_time.dt.hour.astype('uint8')
df['day'] = df.click_time.dt.day.astype('uint8')
test_org_df['click_time'] = pd.to_datetime(test_org_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
test_org_df['hour'] = test_org_df.click_time.dt.hour.astype('uint8')
test_org_df['day'] = test_org_df.click_time.dt.day.astype('uint8')

df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8')
df['minute10'] = (df['minute']/10).astype('uint8') * 10
df['hourminute'] = (df['minute'].astype('uint16') + df['hour'].astype('uint16')*60)
df['hourminute10'] = (df['minute10'].astype('uint16') + df['hour'].astype('uint16')*60)

test_org_df['minute'] = pd.to_datetime(test_org_df.click_time).dt.minute.astype('uint8')
test_org_df['minute10'] = (test_org_df['minute']/10).astype('uint8') * 10
test_org_df['hourminute'] = (test_org_df['minute'].astype('uint16') + test_org_df['hour'].astype('uint16')*60)
test_org_df['hourminute10'] = (test_org_df['minute10'].astype('uint16') + test_org_df['hour'].astype('uint16')*60)


#'ip','app','device','os', 'channel'
patterns = [
'app_day_hourminute',
'device_os_day_hourminute10',
'ip_device_os_day_hourminute10',
]

for ptn in patterns:
    add_col(df,ptn)
