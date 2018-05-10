from sklearn import preprocessing
import pandas as pd
import time
import numpy as np
import sys
import pickle
import os
import gc

nrows=100000
nrows=None

def read_csv(csv_file,df_len=None,nrows=None,usecols=None,dtype=None,is_le=False):
    pkl_file = csv_file[:-4] + '.pkl'
    if os.path.isfile(pkl_file) and nrows == None:
        with open(pkl_file, 'rb') as pk:
            print("loading",pkl_file)
            df = pickle.load(pk)
        if df_len != None and len(df) != df_len:
            print('WARNING:',pkl_file,'is broken')
            df = pd.read_csv(csv_file)
    else:
        print("loading",csv_file)
        df = pd.read_csv(csv_file, nrows=nrows)
    gc.collect()
    return df

def add_col(ptn):
    cum_ptn = "cumcount_" + ptn
    train_df = pd.DataFrame();
    test_df = pd.DataFrame();
    train_df[cum_ptn] = read_csv(work + 'train_' + cum_ptn + '.csv', nrows=nrows)
    test_df[cum_ptn] = read_csv(work + 'test_supplement_' + cum_ptn + '.csv', nrows=nrows)
    cnt_ptn = "count_" + ptn
    train_df[cnt_ptn] = read_csv(work + 'train_' + cnt_ptn + '.csv', nrows=nrows)
    test_df[cnt_ptn] = read_csv(work + 'test_supplement_' + cnt_ptn + '.csv', nrows=nrows)
    all_df = train_df.append(test_df)
    name = 'cumratio_' + ptn
    all_df[name] = round(all_df[cum_ptn]/(all_df[cnt_ptn]-1),4)
    all_df = all_df.fillna(1.1)
    all_df[:len(train_df)][[name]].to_csv(work + 'train_' + name + '.csv', index=False)
    all_df[len(train_df):][[name]].to_csv(work + 'test_supplement_' + name + '.csv', index=False)
    print('done for',work + 'train_' + name + '.csv')
    print('done for',work + 'test_supplement_' + name + '.csv')

############################################
if len(sys.argv) > 1:
    path = '../input' + sys.argv[1] + '/'
    work = '../work' + sys.argv[1] + '/'
else:
    path = '../input/'
    work = '../work/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }


############################################
# read data
############################################
patterns = [
'ip_day',
]

for ptn in patterns:
    add_col(ptn)
