# 1853971 王天 作业二:
#     使用Python实现支持向量机（SVM）分类器（使用自己复现的分类器，不使用sklearn中已实现的SVM分类器）。
#     理解SVM和KNN这两个分类器的差异和权衡之处。
#     将二者应用到CIFAR10图像分类任务上。

# 提交要求：
#     实验报告应包含实现说明、结果截图及运行说明
#     实现代码中应添加必要的注释说明
#     将实验报告及实现代码打包发送到2030809@tongji.edu.cn, 邮件主题为“计算机视觉+学号+姓名+作业二”。

import numpy as np
import pickle

train_data = []
train_label = []
# 加载训练数据集，data_batch_1~data_batch_5中各有10000条训练数据
for i in range(1, 6):
    train_file = open('datasets/data_batch_' + str(i), 'rb')
    data_object = pickle.load(train_file, encoding='bytes')  # 存储为字典格式
    for line in data_object[b'data']:
        train_data.append(line)
    for line in data_object[b'labels']:
        train_label.append(line)

train_data = np.array(train_data).astype("float") # 将数据集中给的整型数据整体转换为浮点型
train_label = np.array(train_label)

# 加载测试集数据, test_batch中有10000条测试数据
test_data = []
test_label = []
test_file = open('datasets/test_batch', 'rb')
data_object = pickle.load(test_file, encoding='bytes')  # 存储为字典格式
for line in data_object[b'data']:
    test_data.append(line)
for line in data_object[b'labels']:
    test_label.append(line)
test_data = np.array(test_data).astype("float") # 将数据集中给的整型数据整体转换为浮点型
test_label = np.array(test_label)

# 数据预处理，在图像分类任务的实践中，对每个特征减去平均值来中心化数据也是非常重要的
train_data = train_data - np.mean(train_data, axis=0)
test_data = test_data - np.mean(test_data, axis=0)
# 对训练集和测试集的每条数据，额外增加一个数值为常量1的维度，用以将W和b两个超参数简化为W一个超参数，即偏差和权重合并
train_data = np.hstack((train_data, np.ones([train_data.shape[0], 1]))) 
test_data = np.hstack((test_data, np.ones([test_data.shape[0], 1])))

class SVM:
    def __init__(self):
        self.W = None

    def loss(self, X, y, reg, delta):
        #   将损失值初始化为0
        loss = 0.0
        dW = np.zeros(self.W.shape)  #  将W的梯度初始化为0
        num = X.shape[0]  # 统计数据集中数据条数
        scores = X.dot(self.W)  #   用训练出来的矩阵计算各个类比的得分值
        correct = scores[range(num), list(y)].reshape(-1, 1)  # 对每条数据根据正确标签作为下标找出正确分类的评分值，再通过reshape（-1，1）转换成1列
        hinge = np.maximum(0, scores - correct + delta)  # 合页函数计算每条数据的损失值
        hinge[range(num), list(y)] = 0  #  正确分类的分数值在上面的运算之后都变成了delta，现在把他们都变成0
        loss = np.sum(hinge) / num + reg * np.sum(self.W * self.W)  #  计算总损失函数值，并且加入了L2正则化防止过拟合
        num_classes = self.W.shape[1]   #   统计分类的类别数目
        #   计算W的梯度
        temp = np.zeros((num, num_classes))
        temp[hinge > 0] = 1
        temp[range(num), list(y)] = 0
        temp[range(num), list(y)] = -np.sum(temp, axis=1)
        dW = (X.T).dot(temp)
        dW = dW / num + 0.5 * reg * self.W
        return loss, dW

    def train(self, X, y, learning_rate=1e-3, reg=1, delta=1, epoch=2000, batch_size=200):
        #   使用随机梯度下降法对SVM进行训练
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  #  y的值从0到分类的类别数目-1，通过此来统计分类的类别数目
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)  #   使用随机数对权重进行初始化
        for it in range(epoch):
            X_batch = None
            y_batch = None
            #   从训练集中随机选择batch_size个数据出来，replace参数表示抽样之后是否放回
            idx_batch = np.random.choice(num_train, batch_size, replace=False)  
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
            loss, grad = self.loss(X_batch, y_batch, reg, delta)    #   计算损失函数值
            self.W -= learning_rate * grad  #   沿梯度方向下降
            if (it + 1) % 100 == 0:   #   每100次迭代显示一次迭代进度
                print('迭代进度 %d / %d: 损失函数值为 %f' % (it + 1, epoch, loss))
        return

    def predict(self, X):
        y = np.zeros(X.shape[0])    #   各个类别的初始得分
        scores = X.dot(self.W)  #   用训练好的W乘以图像数据X即可得到各个类别的得分
        y = np.argmax(scores, axis=1)   #   得分最高的类视为最终识别结果
        return y

svm = SVM()
print("开始训练SVM...")
svm.train(train_data, train_label, learning_rate=1e-7, reg=5e4)    #   训练SVM模型
print("训练完成！")
result= svm.predict(test_data)
#   使用训练好的模型对测试集数据进行预测并统计正确率
print('在测试集上的准确率为: %f' % (np.mean(test_label == result)))
