import pandas as pd
import numpy as np
import sys
import gc
import pytz

nrows=10000
nrows=None
frac=0.01
frac=False

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
train_df = pd.read_csv(input_dir+"/train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], nrows=nrows)
train_df['nextClickLeakDayFlt']  = pd.read_csv(work_dir+"/train_nextClickLeakDayFlt.csv", nrows=nrows)
test_df = pd.read_csv(input_dir+"/test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'], nrows=nrows)
test_df['nextClickLeakDayFlt']  = pd.read_csv(work_dir+"/test_supplement_nextClickLeakDayFlt.csv", nrows=nrows)
if frac:
    train_df = train_df.sample(frac=frac)
    test_df = test_df.sample(frac=frac)
test_df['is_attributed'] = 0

cst = pytz.timezone('Asia/Shanghai')
train_df['click_time'] = pd.to_datetime(train_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
train_df['hour'] = train_df.click_time.dt.hour.astype('uint8')
train_df['day'] = train_df.click_time.dt.day.astype('uint8')
train_df = train_df.reset_index()
test_df['click_time'] = pd.to_datetime(test_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
test_df['hour'] = test_df.click_time.dt.hour.astype('uint8')
test_df['day'] = test_df.click_time.dt.day.astype('uint8')

tgt_hour = (train_df['hour'] >= 12)&(train_df['hour'] <= 22)
tgt_hour_df = train_df[tgt_hour]

fday = train_df.day.min()
lday = train_df.day.max()
print('!'*100)
print('fday:',fday)
print('lday:',lday)
print('!'*100)

def add_col(train_df,test_df,ptn):
    print('start for:',ptn)
    name = "WOEBnd_" + ptn
    dummy = 'is_attributed'
    cols = ptn.split("_")
    cols_with_dummy = cols.copy()
    cols_with_dummy.append(dummy)

    df_list_crr = []
    s = 0.1**8
    for i,day in enumerate(range(fday,lday+1)):
        tdf = tgt_hour_df[tgt_hour_df.day != day]
        gp = tdf[cols_with_dummy].groupby(by=cols)[[dummy]].agg(['count','sum']).is_attributed.reset_index()
        #gp[name] = round(gp['sum']/gp['count'],8)
        pos = tdf[dummy].sum()
        neg = len(tdf) - pos
        gp[name] = np.log((gp['sum']/pos)/((gp['count']-gp['sum']+s)/neg)+1)
        _df =  train_df[train_df.day == day].merge(gp, on=cols, how='left')
        df_list_crr.append(_df)
    _df = pd.concat(df_list_crr)
    _df = _df.fillna(-1).sort_values('index')
    _df[[name]].to_csv(work_dir + '/train_' + name + '.csv', index=False)
    print(work_dir + '/train_' + name + '.csv')

    gp = tgt_hour_df[cols_with_dummy].groupby(by=cols)[[dummy]].agg(['count','sum']).is_attributed.reset_index()
    #gp[name] = round(gp['sum']/gp['count'],8)
    pos = tgt_hour_df[dummy].sum()
    neg = len(tgt_hour_df) - pos
    gp[name] = np.log((gp['sum']/pos)/((gp['count']-gp['sum']+s)/neg)+1)
    _df =  test_df.merge(gp, on=cols, how='left')
    _df = _df.fillna(-1)
    _df[[name]].to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
    print(work_dir + '/test_supplement_' + name + '.csv')
    del _df
    gc.collect()

patterns = [

'ip',
'app',
'device',
'os',
'channel',
'ip_app',
'ip_device',
'ip_os',
'ip_channel',
'app_device',

'app_os',
'app_channel',
'ip_app_device',
'ip_app_os',
'ip_app_channel',
'ip_device_os',
'ip_device_channel',
'ip_os_channel',

'app_device_os',
'app_device_channel',
'app_os_channel',
'ip_app_device_os',
'ip_app_device_channel',
'ip_app_os_channel',
'ip_device_os_channel',
'app_device_os_channel',

'ip_nextClickLeakDayFlt',
'app_nextClickLeakDayFlt',
'device_nextClickLeakDayFlt',
'os_nextClickLeakDayFlt',
'channel_nextClickLeakDayFlt',
'ip_app_nextClickLeakDayFlt',
'ip_device_nextClickLeakDayFlt',

'ip_os_nextClickLeakDayFlt',
'ip_channel_nextClickLeakDayFlt',
'app_device_nextClickLeakDayFlt',
'app_os_nextClickLeakDayFlt',
'app_channel_nextClickLeakDayFlt',
'device_os_nextClickLeakDayFlt',
'device_channel_nextClickLeakDayFlt',
'os_channel_nextClickLeakDayFlt',

]

for ptn in patterns:
    add_col(train_df, test_df, ptn)
