import pandas as pd
import sys
import pytz

#######################
if len(sys.argv) <= 1:
    sys.argv.append("")

input_dir = '../input' + sys.argv[1]
work_dir  = '../work'  + sys.argv[1]

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
    name = "daycount_" + ptn
    dummy = 'is_attributed'
    cols = ptn.split("_")
    cols_with_day = cols.copy()
    tgt = 'day'
    cols_with_day.append(tgt)
    cols_with_dummy = cols_with_day.copy()
    cols_with_dummy.append(dummy)
    gp1 = df[cols_with_dummy].groupby(by=cols_with_day)[[dummy]].count().reset_index().rename(index=str)
    gp2 = gp1[cols_with_day].groupby(by=cols)[[tgt]].count().reset_index().rename(index=str, columns={tgt: name})
    _df = df.merge(gp2, on=cols, how='left')
    _df[[name]][len_train:].to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
    _df[[name]][:len_train].to_csv(work_dir + '/train_' + name + '.csv', index=False)
    print('########### done for: ' + name + ' ###########')
    print(work_dir + '/test_supplement_' + name + '.csv')
    print(work_dir + '/train_' + name + '.csv')

    name = "dayhourcount_" + ptn
    tgt = 'hour'
    cols_with_day.append(tgt)
    cols_with_dummy = cols_with_day.copy()
    cols_with_dummy.append(dummy)
    gp1 = df[cols_with_dummy].groupby(by=cols_with_day)[[dummy]].count().reset_index().rename(index=str)
    gp2 = gp1[cols_with_day].groupby(by=cols)[[tgt]].count().reset_index().rename(index=str, columns={tgt: name})
    _df = df.merge(gp2, on=cols, how='left')
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

#'ip','app','device','os', 'channel'
patterns = [
'ip',
]

for ptn in patterns:
    add_col(df, ptn)
