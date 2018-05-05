from math import sqrt
from dummy_data_generator import Dummy_Data
from utils import sigmoid, logloss

class FTPRL:
    def __init__(self, alpha = 0.1, beta = 0.1, l1 = 0.1, l2 = 0.1):
        if l1 < 0 or l2 < 0:
            raise Exception("Argument Error: Negative Regulizer")

        self._alpha = alpha
        self._beta = beta
        self._l1 = l1
        self._l2 = l2

        self._z = {}
        self._z_bias = 0.0

        self._n = {}
        self._n_bias = 0.0

    def predict(self, features, return_w = False):
        # 即時從z,n產生w
        w = {}
        w_bias = 0.0
        for f_name in features:
            if f_name not in self._z:
                self._z.update({f_name: 0.0})

            if f_name not in self._n:
                self._n.update({f_name: 0.0})

            zx = self._z[f_name]
            nx = self._n[f_name]

            if abs(zx) < self._l1:
                # zx不夠大,被l1 truncate掉
                pass
            else:
                wx = (zx - (1 if zx >= 0 else -1) * self._l1) / (-((self._beta + sqrt(nx)) / self._alpha) + self._l2)
                w.update({f_name: wx})

        if abs(self._z_bias) < self._l1:
            pass
        else:
            w_bias = (self._z_bias - (1 if self._z_bias >= 0 else -1) * self._l1) / (-((self._beta + sqrt(self._n_bias)) / self._alpha) + self._l2)

        wTx = 0.0
        for f_name in features:
            f_value = features[f_name]

            if f_name in w:
                wTx += w[f_name] * f_value
            
        wTx += w_bias

        p = sigmoid(wTx)

        if return_w:
            return p, w, w_bias
        else:
            return p
        

    def update(self, features, label):
        p, w, w_bias = self.predict(features, return_w = True)

        for f_name in features:
            f_value = features[f_name]

            gradient = (p - label) * f_value
            gradient_2 = gradient * gradient
            sigma = (sqrt(self._n[f_name] + gradient_2) - sqrt(self._n[f_name])) / self._alpha

            self._z[f_name] += gradient - sigma * (w[f_name] if f_name in w else 0.0)
            self._n[f_name] += gradient_2

        gradient = p - label
        gradient_2 = gradient * gradient
        sigma = (sqrt(self._n_bias + gradient_2) - sqrt(self._n_bias)) / self._alpha

        self._z_bias += gradient - sigma * w_bias
        self._n_bias += gradient_2

        return  p

if __name__ == "__main__":
    ftprl = FTPRL()
    dummy_data = Dummy_Data()

    loss = 0.0
    iter_num = 0
    for i in range(0, 1000000):
        x, y = dummy_data.generate_data()
        p = ftprl.update(x, y)
        
        loss += logloss(y, p)
        iter_num += 1

        if iter_num % 1000 == 0:
            print("iter:{} logloss: {}".format(iter_num, loss/iter_num))
