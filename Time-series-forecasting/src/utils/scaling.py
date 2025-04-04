from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# scaling: minmax, minmax square, minmax m1p1, standard
def apply_scaling(scaler, trn, val, tst):
    if scaler == 'standard':
        sclr = StandardScaler()
    else:
        sclr = MinMaxScaler()

    trn = sclr.fit_transform(trn)
    val = sclr.fit_transform(val)
    tst = sclr.fit_transform(tst)
    
    return trn, val, tst