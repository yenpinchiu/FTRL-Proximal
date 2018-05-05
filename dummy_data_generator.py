import random
from utils import sigmoid

class Dummy_Data:
    def __init__(self, categorical = 20, numerical = 2, feature_range = 1, weight_range = 100, cate_chance = 0.2):
        if categorical < 0 or numerical < 0 or feature_range < 0 or weight_range < 0:
            raise Exception("Argument Error: Negative Argument")
        
        self._categorical = categorical
        self._numerical = numerical
        self._feature_range = feature_range
        self._weight_range = weight_range
        self._cate_chance = cate_chance

        self._w = {}
        self._w_bias = 0.0

        self._dummy_cate_name = "categorical_{}"
        self._dummy_nume_name = "numerical_{}"


        for i in range(0, self._categorical):
            self._w.update({self._dummy_cate_name.format(i): random.uniform(-self._weight_range, self._weight_range)})

        for i in range(0, self._numerical):
            self._w.update({self._dummy_nume_name.format(i): random.uniform(-self._weight_range, self._weight_range)})

        self._w_bias = random.uniform(-self._weight_range, self._weight_range)

    def generate_data(self):
        dummy_x = {}

        for i in range(0, self._categorical):
            if random.uniform(0, 1) <= self._cate_chance:
                dummy_x.update({self._dummy_cate_name.format(i): 1})

        for i in range(0, self._numerical):
            dummy_x.update({self._dummy_nume_name.format(i): random.uniform(-self._feature_range, self._feature_range)})

        wTx = 0.0
        for x_name in dummy_x:
            if x_name in self._w:
                wTx += self._w[x_name] * dummy_x[x_name]
            
        wTx += self._w_bias

        p = sigmoid(wTx)

        dummy_y = 0
        if random.uniform(0, 1) <= p:
            dummy_y = 1

        return dummy_x, dummy_y

