import numpy as np

expression_names = ["nature", "happy", "confused", "amazing"]
expression_colors = [(0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


class BE2:
    def __init__(self, R1):
        """
        创建对象时，需要输入两个隶属度矩阵
        :param R1: 表情的隶属度矩阵
        """
        self.R1 = R1

    @staticmethod
    def normalized(data):
        """
        保证data中的数据相加为1
        :param data: 输入的一维数组
        :return:
        """
        sum = 0
        for i in range(len(data)):
            sum += data[i]
        if sum == 1:
            return np.array(data)
        return np.array(data) / sum

    @staticmethod
    def min_max_operator(W, R):
        '''
        主因素突出型：M(Λ, V)
        利用最值算子合成矩阵
        :param W:评判因素权向量
        :param R:模糊关系矩阵
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            _list = []
            for row in range(0, np.shape(R)[0]):
                _list.append(min(W[row], R[row, column]))
            B[0, column] = max(_list)
        return B

    @staticmethod
    def min_add_operator(W, R):
        '''
        主因素突出型：M(Λ, +)
        先取小，再求和
        :param W:评判因素权向量
        :param R:模糊关系矩阵
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            _list = []
            for row in range(0, np.shape(R)[0]):
                _list.append(min(W[row], R[row, column]))
            B[0, column] = np.sum(_list)
        return B

    @staticmethod
    def mul_max_operator(W, R):
        '''
        加权平均型：M(*, +)
        利用乘法最大值算子合成矩阵
        :param W:评判因素权向量
        :param R:模糊关系矩阵
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            list = []
            for row in range(0, np.shape(R)[0]):
                list.append(W[row] * R[row, column])
            B[0, column] = max(list)
        return B

    @staticmethod
    def mul_add_operator(W, R):
        '''
        加权平均型：M(*, +)
        先乘再求和
        :param W:评判因素权向量 A = (a1,a2 ,L,an )
        :param R:模糊关系矩阵
        :return:
        '''
        return np.matmul(W, R)

    def get(self, W, R):
        """
        :return: 获得var最大的
        """
        s = [self.normalized(self.min_max_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.min_add_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.mul_max_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.mul_add_operator(W, R).reshape(np.shape(R)[1]))]
        vars = []

        for i in range(len(s)):
            vars.append(s[i].var())

        i = np.argmax(vars)
        return s[i]

    def run(self, W1):
        """

        :param W1: 表情的概率
        :return:
        """
        R = self.get(W1, self.R1),

        return np.dot(R, [1, 0.9, 0.8, 0.6])


def get_expression(marks):
    """
    通过关键点获取表情
    :param marks: 关键点。格式为<br />
    [[1,1] <br />
    [2,2]] <br />
    共 68 个点
    :return: 0  "nature" 1   "happy" 2   "confused" 3 "amazing"
    """
    # 脸的宽度
    face_width = marks[14][0] - marks[0][0]

    # 嘴巴张开程度
    mouth_higth = (marks[66][1] - marks[62][1]) / face_width

    # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
    brow_sum = 0  # 高度之和
    frown_sum = 0  # 两边眉毛距离之和

    # 眼睛睁开程度
    eye_sum = (marks[41][1] - marks[37][1] + marks[40][1] - marks[38][1] +
               marks[47][1] - marks[43][1] + marks[46][1] - marks[44][1])
    eye_hight = (eye_sum / 4) / face_width
    # print("眼睛睁开距离与识别框高度之比：",round(eye_open/self.face_width,3))

    # 头部倾斜程度
    slope = (marks[42][1] - marks[39][1]) / (marks[42][0] - marks[39][0])

    # 两嘴角中间位置占据上下唇高度比例
    center_point = (marks[54][1] + marks[48][1]) / 2
    min_month_point = min([marks[56][1], marks[57][1], marks[58][1]])
    max_month_point = min([marks[50][1], marks[51][1], marks[52][1]])
    mouth_corner_proportion = (center_point - min_month_point) / (max_month_point - min_month_point)

    # 分情况讨论
    # 张嘴，可能是开心或者惊讶
    if mouth_higth >= 0.04:
        if mouth_corner_proportion >= 0.55:
            return 1  # "happy"
        else:
            return 3  # "amazing"

    # 没有张嘴，可能是正常和疑惑
    else:
        if abs(slope) >= 0.3:
            return 2  # "confused"
        else:
            return 0  # "nature"


class AttentionScore():
    def __init__(self):
        expression_membership = np.array([
            [0.7, 0.3, 0.1, 0],  # nature
            [0.1, 0.1, 0.3, 0.5],  # happy
            [0.8, 0.1, 0.1, 0],  # confused
            [0.1, 0.1, 0.4, 0.4]  # amazing
        ])

        # 输入模糊矩阵构建 模糊总和分析 对象
        self.be2 = BE2(expression_membership)

    def __call__(self, *args, **kwargs):
        return self.get_attension_score(*args)

    def get_attension_score(self, landmark, angle):
        """
        获取专注度分数。给定表情和角度，而不是概率的api
        :param landmark: 68个关键点。二维数组形式 68*2
        :param angle: 脸部朝向角度
        :return:0,1,2,3 对应[优良中差]
        """
        expression = get_expression(landmark)

        # 是哪个表情
        expression_weight = np.zeros([4])
        expression_weight[expression] = 1

        angle = abs(angle)
        angle = min([angle, 90])

        if angle <= 20:
            angle_score = 1 - angle / 100
        elif angle <= 40:
            angle_score = 1 - 0.2 - (angle - 20) / 80
        elif angle <= 60:
            angle_score = 1 - 0.2 - 0.25 - (angle - 40) / 60
        else:
            angle_score = 1 - 0.2 - 0.25 - 0.3 - (angle - 60) / 40

        # 表情和转头角度对专注度判断的重要性
        weight = [0.5, 0.5]

        expression_score = self.be2.run(expression_weight)
        return np.dot(weight, [expression_score, angle_score]), expression


attention_score = AttentionScore()  # 默认的专注度识别器


def attention_degrees(face_keypoints, angles):
    return [[*attention_score(fk, ang)] for fk, ang in zip(face_keypoints, angles)]


if __name__ == '__main__':
    attentionScore = AttentionScore()

    marks = np.array([[218.4677, 89.87073],
                      [220.22453, 121.96762],
                      [223.96681, 152.65334],
                      [230.02744, 182.37177],
                      [239.01709, 211.59616],
                      [257.98718, 239.22322],
                      [279.6545, 258.51376],
                      [311.12903, 277.57117],
                      [350.71585, 286.94177],
                      [379.4071, 279.94937],
                      [405.33643, 262.30646],
                      [424.69977, 240.23265],
                      [439.89352, 214.46347],
                      [447.55374, 183.39143],
                      [454.3741, 152.91576],
                      [461.261, 119.00888],
                      [467.35553, 87.80364],
                      [259.43222, 57.652813],
                      [275.41937, 49.985016],
                      [294.87283, 48.28077],
                      [312.70032, 51.04401],
                      [330.32123, 56.870705],
                      [391.8867, 60.823048],
                      [408.12036, 56.251102],
                      [424.36072, 53.19054],
                      [440.7141, 55.756718],
                      [453.48267, 64.0959, ],
                      [360.50037, 94.4313, ],
                      [361.2175, 119.300705],
                      [362.04, 140.86725],
                      [362.67114, 160.99261],
                      [334.13586, 174.79489],
                      [346.0203, 177.28938],
                      [358.7469, 180.27216],
                      [369.87564, 177.55322],
                      [379.49637, 174.3986, ],
                      [277.042, 87.26237],
                      [292.437, 78.486275],
                      [309.88132, 80.89449],
                      [324.76895, 91.85259],
                      [310.34476, 96.826385],
                      [292.4507, 94.96313],
                      [390.42746, 92.99907],
                      [406.47168, 83.25359],
                      [423.17642, 82.80899],
                      [435.07288, 92.21508],
                      [421.45468, 98.3501, ],
                      [405.01886, 98.83813],
                      [310.4157, 218.5501, ],
                      [327.8565, 211.95906],
                      [347.26505, 207.37141],
                      [356.66602, 209.59866],
                      [366.31415, 207.59067],
                      [380.94736, 211.48885],
                      [392.5682, 217.67201],
                      [380.5252, 229.08566],
                      [368.8378, 235.0082, ],
                      [355.32568, 236.70053],
                      [340.88968, 235.58282],
                      [326.3202, 229.22781],
                      [314.68103, 218.55443],
                      [343.84302, 217.35928],
                      [355.64767, 218.21783],
                      [367.20605, 217.28725],
                      [388.59412, 218.09315],
                      [367.44897, 221.60754],
                      [355.48022, 222.92123],
                      [342.7749, 221.94604]])
    for i in range(90):
        print("score:", attentionScore.get_attension_score(marks, i))
