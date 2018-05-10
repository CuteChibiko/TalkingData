import pandas as pd
import sys
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
test_df = pd.read_csv(input_dir+"/test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'], nrows=nrows)
if frac:
    train_df = train_df.sample(frac=frac)
    test_df = test_df.sample(frac=frac)
test_df['is_attributed'] = 0

len_train = len(train_df)
df=train_df.append(test_df)

cst = pytz.timezone('Asia/Shanghai')
df['click_time'] = pd.to_datetime(df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
df['day'] = df.click_time.dt.day.astype('uint8')

fday = df.day.min()
lday = df.day.max()
if len(df[df.day==fday]) < 1000:
    fday += 1
if len(df[df.day==lday]) < 1000:
    lday -= 1
print('!'*100)
print('fday:',fday)
print('lday:',lday)
print('!'*100)
name = 'com_ip'
for d in range(fday,lday+1):
    if d == fday:
        com_set = set(df[df.day==d]['ip'].unique())
    else:
        com_set = com_set & set(df[df.day==d]['ip'].unique())

flt_ip = df.ip.isin(com_set)
df[name] = (df['ip']+1) * flt_ip

df[[name]][len_train:].to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
print('done for',work_dir + '/test_supplement_' + name + '.csv')
df[[name]][:len_train].to_csv(work_dir + '/train_' + name + '.csv', index=False)
print('done for',work_dir + '/train_' + name + '.csv')

######################
com_df = df[flt_ip]
def dump_pct_com_ip(pct):
    name = "com" + str(pct) + "_ip"
    dummy = 'is_attributed'
    cols = ['ip', 'day']
    cols_with_dummy = cols.copy()
    cols_with_dummy.append(dummy)
    gp_ip_day = com_df[cols_with_dummy].groupby(by=cols)[[dummy]].count().reset_index().rename(index=str, columns={dummy: 'count'})
    gp_ip = gp_ip_day[['ip','count']].groupby('ip')[['count']].agg(['mean', 'std'])['count'].reset_index()
    gp_ip['flg'] = (100*gp_ip['std']/gp_ip['mean']) <= pct
    _df = pd.merge(df,gp_ip[['ip','flg']],on='ip',how='left').fillna(False)
    _df[name] = (_df['ip']+1) * _df['flg']

    _df[[name]][len_train:].to_csv(work_dir + '/test_supplement_' + name + '.csv', index=False)
    print('done for',work_dir + '/test_supplement_' + name + '.csv')
    _df[[name]][:len_train].to_csv(work_dir + '/train_' + name + '.csv', index=False)
    print('done for',work_dir + '/train_' + name + '.csv')

dump_pct_com_ip(1)
