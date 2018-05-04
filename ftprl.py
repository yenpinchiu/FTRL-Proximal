from math import log, sqrt
from dummy_data_generator import Dummy_Data
from utils import sigmoid

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
            f_value = features[f_name]

            if f_name not in self._z:
                self._z.update({f_name: 0.0})

            if f_name not in self._n:
                self._n.update({f_name: 0.0})

            zx = self._z[f_name]
            nx = self._n[f_name]

            if zx > self._l1 or zx < -self._l1:
                if zx > self._l1:
                    wx = -(zx - self._l1)
                else:
                    wx = -(zx + self._l1)

                wx /= (((self._beta + sqrt(nx)) / self._alpha) + self._l2)
                
                w.update({f_name: f_value})
            else:
                # zx不夠大,被l1 regulize掉
                pass

        if self._z_bias > self._l1 or self._z_bias < -self._l1:
            if self._z_bias > self._l1:
                w_bias = -(self._z_bias - self._l1)
            else:
                w_bias = -(self._z_bias + self._l1)

            w_bias /= (((self._beta + sqrt(self._n_bias)) / self._alpha) + self._l2)

        wTx = 0.0
        for f_name in features:
            f_value = features[f_name]

            if f_name in w:
                wTx += w[f_name] * f_value
            
        wTx += w_bias

        p = sigmoid(wTx)

        if return_w is True:
            return p, w, w_bias
        else:
            return p
        

    def update(self, features, label):
        p, w, w_bias = self.predict(features, return_w = True)

        for f_name in features:
            f_value = features[f_name]

            gradient = (p - y) * f_value
            gradient_2 = gradient * gradient

            sigma = (sqrt(self._n[f_name] + gradient_2) - sqrt(self._n[f_name])) / self._alpha
            if f_name in w:
                wx = w[f_name]
            else:
                wx = 0.0

            self._z[f_name] += (gradient - (sigma * wx))
            self._n[f_name] += gradient_2

        gradient = p - y
        gradient_2 = gradient * gradient
        sigma = (sqrt(self._n_bias + gradient_2) - sqrt(self._n_bias)) / self._alpha
        self._z_bias += gradient - (sigma * w_bias)
        self._n_bias += gradient_2

        return  p

if __name__ == "__main__":
    ftprl = FTPRL()
    dummy_data = Dummy_Data()

    logloss = 0.0
    update_count = 0
    for i in range(0, 1000):
        x, y = dummy_data.generate_data()
        p = ftprl.update(x, y)
        
        if y == 1:
            logloss -= log(p)
        else:
            logloss -= log(1 - p)

        update_count += 1

        if update_count == 100:
            print(logloss/update_count)
            logloss = 0
            update_count = 0


        
