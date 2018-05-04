from math import exp

def sigmoid(wTx):
    return 1. / (1. + exp(-max(min(wTx, 200.), -200.)))
