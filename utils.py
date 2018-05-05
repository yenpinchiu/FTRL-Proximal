from math import exp, log

def sigmoid(wTx):
    return 1. / (1. + exp(-max(min(wTx, 200.), -200.)))

def logloss(y, p):
    if y == 1:
        return -log(p)
    else:
       return -log(1 - p)