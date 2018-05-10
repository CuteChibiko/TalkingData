import time
import os
import numpy as np
import pandas as pd
import pytz

nrows=100000
nrows=None

work_dir  = '../work'

name = 'nextClickLeakDay'
newName = 'nextClickLeakDayFlt'
train_df = pd.read_csv(work_dir+"/train_" + name + ".csv", nrows=nrows)
test_df = pd.read_csv(work_dir+"/test_supplement_" + name + ".csv", nrows=nrows)
len_train = len(train_df)
df = train_df.append(test_df)

            #d=2
            #if next_click_in_seconds<1800 and next_click_in_seconds>30:
            #    d=1
            #elif next_click_in_seconds<30:
            #    d=0



df[newName] = 2
df[newName] -= (df[name] < 1800) & (df[name] > 30)
df[newName] -= (df[name] < 30) * 2

df[[newName]][len_train:].to_csv(work_dir + '/test_supplement_' + newName + '.csv', index=False)
print(work_dir + '/test_supplement_' + newName + '.csv')
df[[newName]][:len_train].to_csv(work_dir + '/train_' + newName + '.csv', index=False)
print(work_dir + '/train_' + newName + '.csv')
