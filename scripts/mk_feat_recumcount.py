import pandas as pd
import sys
import pytz
import gc

input_dir = '../input'
work_dir  = '../work'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }
nrows=100000
nrows=None
train_df = pd.read_csv(input_dir+"/train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], nrows=nrows).reset_index()
test_df = pd.read_csv(input_dir+"/test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time'], nrows=nrows).reset_index()
train_df['file_id'] = 0
test_df['file_id'] = 1
test_df['is_attributed'] = 0

len_train = len(train_df)
df=train_df.append(test_df)

#######################

def add_col(df,ptn):
    name = "recumcount_" + ptn
    cols = ptn.split("_")
    sub = df[['file_id','index']].copy()
    sub[name] = df.groupby(cols).cumcount()
    tr = sub[sub.file_id == 0].sort_values('index')[[name]]
    te = sub[sub.file_id == 1].sort_values('index')[[name]]
    tr.to_csv(work_dir + '/train_' + name + '.csv', index=False)
    te.to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
    print('########### done for: ' + name + ' ###########')
    print(work_dir + '/train_' + name + '.csv')
    print(work_dir + '/test_supplement_' + name + '.csv')
    del sub,tr,te
    gc.collect()


cst = pytz.timezone('Asia/Shanghai')
df['click_time'] = pd.to_datetime(df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
df['hour'] = df.click_time.dt.hour.astype('uint8')
df['day'] = df.click_time.dt.day.astype('uint8')
df.sort_values(['click_time','index','file_id'], inplace=True, ascending=False)

#'ip','app','device','os', 'channel'
patterns = [
'app_device_os_day',
]

for ptn in patterns:
    add_col(df, ptn)
