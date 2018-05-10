import pandas as pd
import sys
import pytz
import gc
import time

#######################
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
nrows=10000
nrows=None
train_df = pd.read_csv(input_dir+"/train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], nrows=nrows)
test_df = pd.read_csv(input_dir+"/test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'], nrows=nrows)
test_df['is_attributed'] = 0
test_org_df = pd.read_csv(input_dir+"/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'], nrows=nrows)

len_train = len(train_df)
df=train_df.append(test_df)

#######################
def add_col(df,ptn):
    start = time.time()
    print('start for:',ptn)
    name = "var_" + ptn
    cols = ptn.split("_")
    gp = df[cols].groupby(by=cols[0:len(cols)-1])[cols[len(cols)-1]].var().reset_index().rename(index=str, columns={cols[len(cols)-1]: name})
    print(gp.head())
    gp[name] = gp[name].fillna(0).astype(int)
    print(gp.head())
    _df = df.merge(gp, on=cols[0:len(cols)-1], how='left')
    _df[[name]][len_train:].to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
    _df[[name]][:len_train].to_csv(work_dir + '/train_' + name + '.csv', index=False)
    print('########### done for: ' + name + ' ###########', time.time()-start/60)
    print(work_dir + '/test_supplement_' + name + '.csv')
    print(work_dir + '/train_' + name + '.csv')
    del _df
    gc.collect()

cst = pytz.timezone('Asia/Shanghai')
df['click_time'] = pd.to_datetime(df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
df['hour'] = df.click_time.dt.hour.astype('uint8')
df['day'] = df.click_time.dt.day.astype('uint8')
test_org_df['click_time'] = pd.to_datetime(test_org_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
test_org_df['hour'] = test_org_df.click_time.dt.hour.astype('uint8')
test_org_df['day'] = test_org_df.click_time.dt.day.astype('uint8')

#'ip','app','device','os', 'channel'
patterns = [
'ip_device_hour',
]

for ptn in patterns:
    add_col(df, ptn)
