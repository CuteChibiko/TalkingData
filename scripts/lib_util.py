import sys

# this script makes it possible to describe a model in one argument.

default_target = 'model=LGBM_feat=lgbmBest_categoricalThreVal=10000_validation=subm_params=-,gbdt,0.45,0.04,188,7,auc,20,5,0,70,76,binary,0,0,32.0,1.0,200000,1,0'
target = ''

def get_opt(name,default=None):
    global target
    if target == '':
        target = get_target()
    if target == '':
        return default
    flds = target.replace('__','').split('_')
    for fld in flds:
        if fld == '':
            continue
        key, val = fld.split('=')
        if key == name:
            if isinstance(default, int):
                val = int(val)
            elif isinstance(default, float):
                val = float(val)
            else:
                val = val.replace('','_')
            return val
    return default

def get_target():
    global target
    if target != '':
        return target
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = default_target
    return target

def reset_target():
    global target
    target = ''
    a = get_target()
    return get_target()

def set_target(tgt):
    global target
    target = tgt
