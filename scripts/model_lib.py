# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random as rn
from sklearn.metrics import roc_auc_score
import sys
import os
from lib_util import get_target,get_opt
import lightgbm as lgb
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate, BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
import gc

def get_params(params_str):
    if get_opt('model') == 'keras':
        names = ['batch_size', 'dense_cate', 'dense_nume_n_layers', 'drop', 'emb_cate', 'epochs_for_lr', 'lr', 'lr_fin', 'lr_init', 'max_epochs', 'n_layers', 'patience']
    elif 'LGBM' in get_opt('model'):
        names = ['boosting_type','colsample_bytree','learning_rate','max_bin','max_depth','metric','min_child_samples','min_child_weight','min_split_gain','nthread','num_leaves','objective','reg_alpha','reg_lambda','scale_pos_weight','subsample','subsample_for_bin','subsample_freq','verbose']
    else:
        print("no valid target")
        sys.exit(1)
    pvals = params_str.split(',')
    del pvals[0]
    if len(pvals) != len(names):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ERR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('params: count is not fit',len(pvals), len(names))
        print('params_str:',params_str)
        print('names:',names)
        print('param_values:',pvals)
        sys.exit()
    params = dict(zip(names, pvals))
    return params

def LGBM(X_tr,X_va,X_te,predictors,cat_feats,seed=2018):
    params_str = get_opt('params')
    if params_str != None:
        params = get_params(params_str)
        return LGBM_helper(X_tr,X_va,X_te,predictors,cat_feats,params,seed=2018)

def Keras(X_tr,X_va,X_te,predictors,cat_feats,seed=2018):
    params_str = get_opt('params')
    if params_str != None:
        params = get_params(params_str)
        return Keras0_helper(X_tr,X_va,X_te,predictors,cat_feats,params,seed=2018)

def LGBM_helper(_X_tr,_X_va,_X_te,predictors,cat_feats,params,seed=2018):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    X_tr = _X_tr[predictors]
    X_va = _X_va[predictors]
    X_te = _X_te[predictors]
    y_tr = _X_tr['is_attributed']
    y_va = _X_va['is_attributed']
    y_te = _X_te['is_attributed']
    params['feature_fraction_seed'] = seed
    params['bagging_seed'] = seed
    params['drop_seed'] = seed
    params['data_random_seed'] = seed
    params['num_leaves'] = int(params['num_leaves'])
    params['subsample_for_bin'] = int(params['subsample_for_bin'])
    params['max_depth'] = int(np.log2(params['num_leaves'])+1.2)
    params['max_bin'] = int(params['max_bin'])
    print('*'*50)
    for k,v in sorted(params.items()):
        print(k,':',v)
    columns = X_tr.columns

    print('start for lgvalid')
    lgvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_feats)
    _X_va.drop(predictors,axis=1)
    del _X_va, X_va, y_va
    gc.collect()

    print('start for lgtrain')
    lgtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_feats)
    _X_te.drop(predictors,axis=1)
    del _X_tr, X_tr, y_tr
    gc.collect()

    evals_results = {}
    if get_opt('trainCheck','-') == 'on':
         valid_names=['train','valid']
         valid_sets=[lgtrain, lgvalid]
    else:
         valid_names=['valid']
         valid_sets=[lgvalid]
    if get_opt('testCheck','-') == 'on':
         valid_names.append('test')
         lgtest = lgb.Dataset(X_te, label=y_te, categorical_feature=cat_feats)
         valid_sets.append(lgtest)

    print('start training')
    bst = lgb.train(params,
                     lgtrain,
                     valid_sets=valid_sets,
                     valid_names=valid_names,
                     evals_result=evals_results,
                     num_boost_round=2000,
                     early_stopping_rounds=100,
                     verbose_eval=10,
                     )

    importance = bst.feature_importance()
    print('importance (count)')
    tuples = sorted(zip(columns, importance), key=lambda x: x[1],reverse=True)
    for col, val in tuples:
        print(val,"\t",col)

    importance = bst.feature_importance(importance_type='gain')
    print('importance (gain)')
    tuples = sorted(zip(columns, importance), key=lambda x: x[1],reverse=True)
    for col, val in tuples:
        print(val,"\t",col)

    n_estimators = bst.best_iteration
    metric = params['metric']
    auc = evals_results['valid'][metric][n_estimators-1]
    _X_te['pred'] = bst.predict(X_te)

    return auc

class EarlyStopping(Callback):
    def __init__(self,training_data=False,validation_data=False, testing_data=False, min_delta=0, patience=0, model_file=None, verbose=0):
        super(EarlyStopping, self).__init__()
        self.best_epoch = 0
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        if training_data:
            self.x_tr = training_data[0]
            self.y_tr = training_data[1]
        else:
            self.x_tr = False
            self.y_tr = False
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        if testing_data:
            self.x_te = testing_data[0]
            self.y_te = testing_data[1]
        else:
            self.x_te = False
            self.y_te = False
        self.model_file = model_file
    def on_train_begin(self, logs={}):
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ',self.best_epoch,': EarlyStopping')
    def on_epoch_end(self, epoch, logs={}):

        if self.x_tr:
            y_pred = self.model.predict(self.x_tr,batch_size=100000)
            roc_tr = roc_auc_score(self.y_tr, y_pred)
        else:
            roc_tr = 0

        y_hat_val=self.model.predict(self.x_val,batch_size=100000)
        roc_val = roc_auc_score(self.y_val, y_hat_val)

        if self.x_te:
            y_hat_te=self.model.predict(self.x_te,batch_size=100000)
            roc_te = roc_auc_score(self.y_te, y_hat_te)
        else:
            roc_te = 0
        print('roc-auc: %s - roc-auc_val: %s - roc-auc_test: %s' % (str(round(roc_tr,6)),str(round(roc_val,6)), str(round(roc_te,6))),end=100*' '+'\n')

        if self.model_file:
            print("saving",self.model_file+'.'+str(epoch))
            self.model.save_weights(self.model_file+'.'+str(epoch))
        if(self.x_val):
            if get_opt('testCheck','-') == 'on': 
                current = roc_te
            else:
                current = roc_val
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.best_epoch = epoch
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True

def Keras0_helper(_X_tr,_X_va,_X_te,predictors,cat_feats,params,seed=2018):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    X_tr = _X_tr[predictors]
    X_va = _X_va[predictors]
    X_te = _X_te[predictors]
    y_tr = _X_tr['is_attributed']
    y_va = _X_va['is_attributed']
    y_te = _X_te['is_attributed']
    print('*************params**************')
    for f in sorted(params): print(f+":",params[f])
    batch_size = int(params['batch_size'])
    epochs_for_lr = float(params['epochs_for_lr'])
    max_epochs = int(params['max_epochs'])
    emb_cate = int(params['emb_cate'])
    dense_cate = int(params['dense_cate'])
    dense_nume_n_layers = int(params['dense_nume_n_layers'])
    drop = float(params['drop'])
    lr= float(params['lr'])
    lr_init = float(params['lr_init'])
    lr_fin = float(params['lr_fin'])
    n_layers = int(params['n_layers'])
    patience = int(params['patience'])
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    input_list = []
    emb_list = []
    numerical_feats = []
    tot_emb_n = 0
    for col in X_tr:
        if col not in cat_feats:
            numerical_feats.append(col)
    if len(cat_feats) > 0:
        for col in cat_feats:
            train_dict[col] = np.array(X_tr[col])
            valid_dict[col] = np.array(X_va[col])
            test_dict[col] = np.array(X_te[col])
            inpt = Input(shape=[1], name = col)
            input_list.append(inpt)
            max_val = np.max([X_tr[col].max(), X_va[col].max(), X_te[col].max()])+1
            emb_n = np.min([emb_cate, max_val])
            if get_opt('fixEmb','on') == 'on':
                emb_n = emb_cate
            tot_emb_n += emb_n
            if emb_n == 1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Warinig!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! emb_1 = 1")
                return 0
            print('Embedding size:',max_val, emb_cate, X_tr[col].max(), X_va[col].max(), X_te[col].max(), emb_n,col)
            embd = Embedding(max_val, emb_n)(inpt)
            emb_list.append(embd)
        if len(emb_list) == 1:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Warinig!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! emb_list = 1")
            return 0
        fe = concatenate(emb_list)
        s_dout = SpatialDropout1D(drop)(fe)
        x1 = Flatten()(s_dout)

    if get_opt('sameNDenseAsEmb','-') == 'on':
        dense_cate = tot_emb_n
    if len(numerical_feats) > 0:
        train_dict['numerical'] = X_tr[numerical_feats].values
        valid_dict['numerical'] = X_va[numerical_feats].values
        test_dict['numerical'] = X_te[numerical_feats].values
        inpt = Input((len(numerical_feats),),name='numerical')
        input_list.append(inpt)
        x2 = inpt
        for n in range(dense_nume_n_layers):
            x2 = Dense(dense_cate,activation='relu',kernel_initializer=RandomUniform(seed=seed))(x2)
            if get_opt('numeDropout','on') != 'off':
                x2 = Dropout(drop)(x2)
            if get_opt('NumeBatchNormalization','on') != 'off':
                x2 = BatchNormalization()(x2)

    if len(numerical_feats) > 0 and len(cat_feats) > 0:
        x = concatenate([x1, x2])
    elif len(numerical_feats) > 0:
        x =  x2
    elif len(cat_feats) > 0:
        x =  x1
    else:
        return 0 # for small data test

    for n in range(n_layers):
        x = Dense(dense_cate,activation='relu',kernel_initializer=RandomUniform(seed=seed))(x)
        if get_opt('lastDropout','on') != 'off':
            x = Dropout(drop)(x)
        if get_opt('BatchNormalization','off') == 'on' or get_opt('LastBatchNormalization','off') == 'on':
            x = BatchNormalization()(x)
    outp = Dense(1,activation='sigmoid',kernel_initializer=RandomUniform(seed=seed))(x)
    model = Model(inputs=input_list, outputs=outp)
    if get_opt('optimizer','expo') == 'adam':
        optimizer = Adam(lr=lr)
    elif get_opt('optimizer','expo') == 'nadam':
        optimizer = Nadam(lr=lr)
    else:
        exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
        steps = int(len(X_tr) / batch_size) * epochs_for_lr
        lr_init, lr_fin = 0.001, 0.0001
        lr_decay = exp_decay(lr_init, lr_fin, steps)
        optimizer = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='binary_crossentropy',optimizer=optimizer)
    model.summary()
    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png')

    model_file = '../work/weights.'+str(os.getpid())+'.hdf5'
    if get_opt('trainCheck','-') == 'on': 
        training_data=(train_dict, y_tr)
    else:
        training_data=False
    if get_opt('testCheck','-') == 'on':
        testing_data=(test_dict, y_te)
    else:
        testing_data=False
    aucEarlyStopping = EarlyStopping(
        training_data=training_data,
        validation_data=(valid_dict,y_va),
        testing_data=testing_data,
        patience=patience,
        model_file=model_file,
        verbose=1)
    model.fit(train_dict,
        y_tr,
        validation_data=[valid_dict, y_va],
        batch_size=batch_size,
        epochs=max_epochs,
        shuffle=True,
        verbose=2,
        callbacks=[aucEarlyStopping])
    best_epoch = aucEarlyStopping.best_epoch
    print('loading',model_file+'.'+str(best_epoch))
    model.load_weights(model_file+'.'+str(best_epoch))
    _X_te['pred'] = model.predict(test_dict, batch_size=batch_size, verbose=2)[:,0]
    _X_va['pred'] = model.predict(valid_dict, batch_size=batch_size, verbose=2)[:,0]
    if get_opt('avgEpoch',0) > 0:
        added = 1
        for i in range(min(get_opt('avgEpoch',0),patience)):
            best_epoch = aucEarlyStopping.best_epoch + (i+1)
            if best_epoch >= max_epochs:
                continue
            print('loading',model_file+'.'+str(best_epoch))
            model.load_weights(model_file+'.'+str(best_epoch))
            _X_te['pred'] += model.predict(test_dict, batch_size=batch_size, verbose=2)[:,0]*0.5
            _X_va['pred'] += model.predict(valid_dict, batch_size=batch_size, verbose=2)[:,0]*0.5
            added += 0.5
            best_epoch = aucEarlyStopping.best_epoch - (i+1)
            if best_epoch < 0:
                continue
            print('loading',model_file+'.'+str(best_epoch))
            model.load_weights(model_file+'.'+str(best_epoch))
            _X_te['pred'] += model.predict(test_dict, batch_size=batch_size, verbose=2)[:,0]*0.5
            _X_va['pred'] += model.predict(valid_dict, batch_size=batch_size, verbose=2)[:,0]*0.5
            added += 0.5
        _X_te['pred'] /= added
        _X_va['pred'] /= added

    os.system('rm -f '+model_file+'.*')
    auc = roc_auc_score(y_va, _X_va.pred)
    return auc

def Predict(X_tr,X_va,X_te,predictors,cat_feats,seed=2018):
    model = get_opt('model')
    if 'LGBM' in model:
        return LGBM(X_tr,X_va,X_te,predictors,cat_feats,seed=2018)
    elif 'keras' in model:
        return Keras(X_tr,X_va,X_te,predictors,cat_feats,seed=2018)
    else:
        print("no valid model")
        sys.exit(1)
