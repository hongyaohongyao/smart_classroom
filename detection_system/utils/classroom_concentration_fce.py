import math

import numpy as np


class ClassroomConcentrationFCE:
    def __init__(self):
        self.v = np.array([5, 4, 3, 2, 1])  # 评价等级
        self.R = []
        self.W = None  # 一级指标权重

        self.factor_name = []
        pass

    def register_new_factor(self, name, R):
        self.R.append(R)
        self.factor_name.append(name)
        pass

    def get_concentration(self, factors):
        B = []
        for i, name in enumerate(self.factor_name):
            w2 = factors[name]
            B.append(np.matmul(w2, self.R[i]))
        B = np.stack(B, axis=0)
        D = np.matmul(self.W, B)
        return np.matmul(D, self.v)

    def set_weights_of_factor(self, counts):
        self.W = np.empty(len(self.factor_name), dtype=np.float)  # 一级指标权重
        for i, name in enumerate(self.factor_name):
            count = counts[name]
            if isinstance(count, list):
                count = np.array(count)
            self.W[i] = self.factor_info_entropy(count)
        self.W = self.softmax(self.W)

    @staticmethod
    def factor_info_entropy(f):
        delta = 1e-7  # 添加一个微小值可以防止负无限大(np.log(0))的发生。
        return np.sum(f * np.log(f + delta)) - math.log(1 / len(f))

    @staticmethod
    def softmax(x):
        """ softmax function """
        exp = np.exp(x)
        x = exp / np.sum(exp, axis=1, keepdims=True)
        return x


if __name__ == '__main__':
    x = np.random.randint(low=1, high=5, size=(2, 3))  # 生成一个2x3的矩阵，取值范围在1-5之间
    print("原始 ：\n", x)

    x_ = ClassroomConcentrationFCE.softmax(x)
    print("变换后 ：\n", x_)
