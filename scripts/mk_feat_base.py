import pandas as pd
import sys
import pytz

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
test_df = pd.read_csv(input_dir+"/test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time'], nrows=nrows)
test_df['is_attributed'] = 0

cst = pytz.timezone('Asia/Shanghai')
def set_day_hour(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
    df['hour'] = df.click_time.dt.hour.astype('uint8')
    df['day'] = df.click_time.dt.day.astype('uint8')
    del df['click_time']

set_day_hour(train_df)
set_day_hour(test_df)

train_df.to_csv(work_dir + '/train_base.csv',index=False)
test_df.to_csv(work_dir + '/test_supplement_base.csv',index=False)

for fld in ['ip','app','device','os', 'channel','day','hour', 'is_attributed']:
    train_df[[fld]].to_csv(work_dir + '/train_base_' + fld + '.csv',index=False)
    test_df[[fld]].to_csv(work_dir + '/test_supplement_base_' + fld + '.csv',index=False)
