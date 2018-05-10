#-*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import sys
import pickle
import os
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.special import logit
from lib_util import get_target,get_opt,set_target,reset_target
import shutil
import pdb

target=get_target()
nrows=get_opt('nrows',-1)
if nrows == -1:
    nrows=None
path = '../input/'
work = '../work/'
csv_dir='../csv/'
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

# wrapper of pd.read_csv with cache
def read_csv(csv_file,df_len=None,nrows=None,usecols=None,dtype=None):
    pkl_file = csv_file[:-4] + '.pkl'
    if os.path.isfile(pkl_file) and nrows == None:
        with open(pkl_file, 'rb') as pk:
            print("loading",pkl_file)
            df = pickle.load(pk)
        if df_len != None and len(df) != df_len:
            print('ERROR!!!!!!!!!!!!!!!!!!!!!!!',pkl_file,'is broken',len(df),df_len)
            sys.exit(1)
    else:
        print("loading",csv_file)
        df = pd.read_csv(csv_file, nrows=nrows)
        if 'next' in csv_file:
            df = np.absolute(df)
        for ptn in df:
            if dtype:
                df = df.astype(dtype)
            else:
                df[ptn] = df[ptn].astype(get_type_with_fld_check(df,ptn))
        if nrows == None and (df_len == None or len(df) == df_len):
            print("saving cache file",pkl_file)
            with open(pkl_file+str(os.getpid()), 'wb') as pk:
                pickle.dump(df,pk,protocol=4)
            shutil.move(pkl_file+str(os.getpid()), pkl_file)
        if nrows == None and df_len and len(df) != df_len:
            print('ERROR!!!!!!!!!!!!!!!!!!!!!!!',csv_file,'is broken')
            sys.exit(1)
    if usecols != None:
        df = df[usecols]
    if nrows != None:
        df = df[:nrows]
    gc.collect()
    if df_len != None and len(df) != df_len:
        print('ERROR!!!!!!!!!!!!!!!!!!!!!!!',csv_file,'line is not same',df_len,len(df))
        sys.exit(1)
    return df

def get_type_with_fld_check(df,ptn):
    max_val = df[ptn].max()
    if  'cumratio' in ptn or 'mean_' in ptn or 'Ratio' in ptn or 'CVR' in ptn or 'WOE' in ptn:
        dtype = 'float16'
    else:
        if max_val < 256:
            dtype = 'uint8'
        elif max_val < 65536:
            dtype = 'uint16'
        else:
            dtype = 'uint32'
    return dtype

def get_type(df,ptn):
    max_val = df[ptn].max()
    if max_val < 256:
        dtype = 'uint8'
    elif max_val < 65536:
        dtype = 'uint16'
    else:
        dtype = 'uint32'
    return dtype


def read_data_ph1():
    keep_patterns = []
    feat_opt = get_opt('feat','none')

    if 'lgbmBest' == feat_opt:
        numerical_patterns = ['WOEBnd_ip_nextClickLeakDayFlt', 'WOEBnd_app_nextClickLeakDayFlt', 'WOEBnd_device_nextClickLeakDayFlt', 'WOEBnd_os_nextClickLeakDayFlt', 'WOEBnd_channel_nextClickLeakDayFlt', 'WOEBnd_ip_app_nextClickLeakDayFlt', 'WOEBnd_ip_device_nextClickLeakDayFlt', 'WOEBnd_ip_os_nextClickLeakDayFlt', 'WOEBnd_ip_channel_nextClickLeakDayFlt', 'WOEBnd_app_device_nextClickLeakDayFlt', 'WOEBnd_app_os_nextClickLeakDayFlt', 'WOEBnd_app_channel_nextClickLeakDayFlt', 'WOEBnd_device_os_nextClickLeakDayFlt', 'WOEBnd_device_channel_nextClickLeakDayFlt', 'WOEBnd_os_channel_nextClickLeakDayFlt', 'WOEBnd_ip', 'WOEBnd_app', 'WOEBnd_device', 'WOEBnd_os', 'WOEBnd_channel', 'WOEBnd_ip_app', 'WOEBnd_ip_device', 'WOEBnd_ip_os', 'WOEBnd_ip_channel', 'WOEBnd_app_device', 'WOEBnd_app_os', 'WOEBnd_app_channel', 'WOEBnd_ip_app_device', 'WOEBnd_ip_app_os', 'WOEBnd_ip_app_channel', 'WOEBnd_ip_device_os', 'WOEBnd_ip_device_channel', 'WOEBnd_ip_os_channel', 'WOEBnd_app_device_os', 'WOEBnd_app_device_channel', 'WOEBnd_app_os_channel', 'WOEBnd_ip_app_device_os', 'WOEBnd_ip_app_device_channel', 'WOEBnd_ip_app_os_channel', 'WOEBnd_ip_device_os_channel', 'WOEBnd_app_device_os_channel', 'countRatio_ip_machine', 'countRatio_ip_channel', 'countRatio_machine_ip', 'countRatio_app_channel', 'countRatio_channel_app', 'uniqueCount_day_ip_os', 'uniqueCount_day_ip_device', 'uniqueCountRatio_day_ip_channel', 'uniqueCount_day_ip_machine', 'uniqueCount_day_ip_app',  'uniqueCount_machine_app', 'uniqueCount_machine_channel', 'uniqueCount_machine_ip', 'nextClickLeakDay', 'nextNextClickLeakDay', 'dayhourminute10count_ip_device_os', 'dayhourminute10count_ip_channel', 'dayhourminute10count_app_os_channel', 'cumratio_ip_day', 'cumcount_ip_day', 'count_ip_os', 'count_ip_device_os_day_hourminute10', 'count_ip_app_os_channel_day', 'count_ip_app_os_channel', 'count_ip_app_device_os_day_hour', 'count_ip_app_device_day', 'count_ip_app_device_channel_day', 'count_ip', 'count_device_os_day_hourminute10', 'count_app_os_channel_day_hour', 'count_app_device_day_hour', 'count_app_device_channel_day_hour', 'recumcount_app_device_os_day', 'var_ip_device_hour', 'count_app_day_hourminute']
        cat_patterns = ['cat_os', 'cat_hour', 'cat_device', 'cat_dayhourcount_ip', 'cat_com1_ip', 'cat_channel', 'cat_app']
    elif 'kerasBest' == feat_opt:
        numerical_patterns = ['uniqueCountRatio_day_ip_machine', 'uniqueCountRatio_day_ip_app', 'uniqueCountRatio_day_ip_channel', 'uniqueCount_day_ip_machine', 'uniqueCount_day_ip_app', 'uniqueCount_day_ip_channel', 'uniqueCount_machine_app', 'uniqueCount_machine_channel', 'uniqueCount_machine_ip', 'nextClickLeakDay', 'dayhourcount_ip', 'count_ip', 'count_ip_app_device_os_day_hour', 'count_app_channel', 'cumcount_ip_app_device_os_day_hour', 'count_device_os_day_hourminute10', 'count_app_device_day_hour', 'dayhourminute10count_ip']
        cat_patterns = ['cat_nextClickLeakDay', 'cat_nextNextClickLeakDay', 'cat_app', 'cat_device', 'cat_os', 'cat_count_ip', 'cat_count_app_channel', 'cat_hour', 'cat_dayhourcount_ip']
    else:
        print('ERR: no valid feat !!!!!!!!!!!!!!!!')
        sys.exit(1)

    print("start reading feature for",feat_opt)

    # all cache
    tgt = 'model=' + get_opt('model','none')
    tgt += '_nrows=' + get_opt('nrows','0') 
    tgt += '_feat=' + get_opt('feat','0') 
    tgt += '_categoricalThreVal=' + get_opt('categoricalThreVal','1000') 
    tgt += '_offlineADD=' + get_opt('offlineADD','off') 
    tgt += '_sample=' + get_opt('sample','0.0') 
    tgt += '_noTestSample=' + get_opt('noTestSample','off') 
    tgt += '_noLogDev=' + get_opt('noLogDev','off') 
    tgt += '_smallTest=' + get_opt('smallTest','off') 
    tgt += '_ver=3'
    tr_pkl_file = '../work/train_' + tgt + '.pkl'
    te_pkl_file = '../work/test_supplement_' + tgt + '.pkl'
    if os.path.isfile(tr_pkl_file) == True and os.path.isfile(te_pkl_file) == True:
        with open(tr_pkl_file, 'rb') as pk:
            print("loading",tr_pkl_file)
            train_df = pickle.load(pk)
        with open(te_pkl_file, 'rb') as pk:
            print("loading",te_pkl_file)
            test_df = pickle.load(pk)
        gc.collect()
        return train_df, test_df, numerical_patterns, cat_patterns

    # reading base data
    train_df = read_csv(work+"train_base.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel','day','hour','is_attributed'],nrows=nrows)
    test_df = read_csv(work+"test_supplement_base.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel','day','hour'],nrows=nrows)
    test_df['is_attributed'] = 0

    # reading numerical data
    n = 0
    for ptn in numerical_patterns:
        n+=1
        print('start for',ptn,n,'/',len(numerical_patterns))
        if ptn in train_df.columns: continue
        train_df[ptn] = read_csv(work + 'train_' + ptn + '.csv', nrows=nrows, df_len=len(train_df))
        test_df[ptn] = read_csv(work + 'test_supplement_' + ptn + '.csv', nrows=nrows, df_len=len(test_df))
    
    #reading categorical data
    n = 0
    for ptn in cat_patterns:
        n+=1
        print('start categorical convert for',ptn,n,'/',len(cat_patterns))
        if ptn in train_df.columns:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! warning cat ptn is in train_df.columns !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(ptn,train_df.columns)
        org_ptn = ptn[4:]
        if org_ptn in train_df.columns:
            _train_df = train_df[[org_ptn]]
            _test_df = test_df[[org_ptn]]
        else:
            _train_df = read_csv(work + 'train_' + org_ptn + '.csv', nrows=nrows, df_len=len(train_df))
            _test_df = read_csv(work + 'test_supplement_' + org_ptn + '.csv', nrows=nrows, df_len=len(test_df))
        _train_df = _train_df.rename(columns={org_ptn: ptn})
        _test_df = _test_df.rename(columns={org_ptn: ptn})
        
        len_train = len(_train_df)
        _df = _train_df.append(_test_df)
        thre_val = get_opt('categoricalThreVal',1000)
        max_val = _df[ptn].max()
        if 'cat_device' == ptn and get_opt('noLogDev','-') == 'on':
            _df[ptn] = LabelEncoder().fit_transform(_df[ptn])
        elif thre_val > 0 and max_val > thre_val:
            if 'cumratio' in ptn:
                fixed_vals = (10000*df[ptn]).astype('uint16')
            else:
                fixed_vals = (np.log2(_df[ptn]+1)*thre_val/100).astype('uint16')
            _df[ptn] = LabelEncoder().fit_transform(fixed_vals)
            print('logged for',ptn,max_val,fixed_vals.max(), _df[ptn].max())
        else:
            _df[ptn] = LabelEncoder().fit_transform(_df[ptn])
        _df[ptn] = _df[ptn].astype(get_type(_df,ptn))

        train_df[ptn] = _df[:len_train]
        test_df[ptn] = _df[len_train:]
        gc.collect()

    # numerical data conversion
    for ptn in numerical_patterns:
        if get_opt('model','-') == 'keras':
            print('start for numerical convert',ptn)
            all_df = train_df[[ptn]].append(test_df[[ptn]])
            if 'cumratio' in ptn or 'CVRTgt' in ptn or 'WOETgt' in ptn:
                pass
            else:
                all_df = np.log2(all_df+1)
            all_df = StandardScaler().fit_transform(all_df).astype('float16')
            train_df[ptn] = all_df[:len(train_df)]
            test_df[ptn] = all_df[len(train_df):]

    # saving cache
    print("saving",tr_pkl_file)
    with open(tr_pkl_file+str(os.getpid()), 'wb') as pk:
        pickle.dump(train_df,pk,protocol=4)
    shutil.move(tr_pkl_file+str(os.getpid()), tr_pkl_file)
    print("saving",te_pkl_file)
    with open(te_pkl_file+str(os.getpid()), 'wb') as pk:
        pickle.dump(test_df,pk,protocol=4)
    shutil.move(te_pkl_file+str(os.getpid()), te_pkl_file)
    print('saved cache file')

    gc.collect()
    return train_df, test_df, numerical_patterns, cat_patterns
