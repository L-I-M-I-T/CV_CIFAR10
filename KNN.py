# 1853971 王天 作业二:
#     使用Python实现K最近邻（KNN）分类器（使用自己复现的分类器，不使用sklearn中已实现的KNN分类器）。
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
ratio1 = 50
# 加载训练数据集，data_batch_1~data_batch_5中各有10000条训练数据，由于KNN在检测时效率很低，因此只取训练数据的一小部分(规模缩小为1/ratio1)
for i in range(1, 6):
    train_file = open('datasets/data_batch_' + str(i), 'rb')
    data_object = pickle.load(train_file, encoding='bytes')  # 存储为字典格式
    count = 0
    for line in data_object[b'data']:
        count += 1
        if (count % ratio1 == 0):
            train_data.append(line)
    count = 0
    for line in data_object[b'labels']:
        count += 1
        if (count % ratio1 == 0):
            train_label.append(line)

train_data = np.array(train_data).astype("float") # 将数据集中给的整型数据整体转换为浮点型
train_label = np.array(train_label)

ratio2 = 100
# 加载测试集数据，test_batch中有10000条测试数据，由于KNN在检测时效率很低，因此只取测试数据的一小部分(规模缩小为1/ratio2)
test_data = []
test_label = []
test_file = open('datasets/test_batch', 'rb')
data_object = pickle.load(test_file, encoding='bytes')  # 存储为字典格式
count = 0
for line in data_object[b'data']:
    count += 1
    if (count % ratio2 == 0):
        test_data.append(line)
count = 0
for line in data_object[b'labels']:
    count += 1
    if (count % ratio2 == 0):
        test_label.append(line)
test_data = np.array(test_data).astype("float") # 将数据集中给的整型数据整体转换为浮点型
test_label = np.array(test_label)

# 数据预处理，在图像分类任务的实践中，对每个特征减去平均值来中心化数据也是非常重要的
train_data = train_data - np.mean(train_data, axis=0)
test_data = test_data - np.mean(test_data, axis=0)

def takefirst(elem):    #   返回多元组中的第一个元素
    return elem[0]

def find(arr, value):   #   查找数组中value数值第一次出现的位置
    for i in range(0, len(arr)):
        if (value == arr[i]):
            return i

class KNN:
    def __init__(self):
        self.W = None

    def predict(self, X1, X2, Y, K = 5, batch_size = 200):
        y = []
        for i in range(0,X1.shape[0]):  #   对测试集中的各条数据进行分类测试
            if ((i % 10) ==0):  #   每测试10条显示一次测试进度
                print("测试进度为: %f " % (i/X1.shape[0]))
            res = []
            for j in range(0, X2.shape[0]): #   遍历训练集中的各条数据，使用L2距离对比相似程度
                dist=0
                for k in range(0,X1.shape[1]):  #   计算两张图像（一张测试集，一张训练集）的L2距离
                    dist += ((X1[i][k] - X2[j][k]) ** 2) ** 0.5
                res.append([dist,Y[j]])
            score = []
            for j in range (0, np.max(Y) + 1):  #   初始化各个类别在最近K邻中的出现次数
                score.append(0) 
            for k in range(0,K):    #找到最近K邻的数据
                MIN = 10000000
                temp = 0
                for l in range(0,X2.shape[0]):
                    if (res[l][0] < MIN):
                        MIN = res[l][0]
                        temp = l
                score[res[temp][1]] += 1
                res[temp][0] = 10000000   
            y.append(find(score, np.max(score)))    #   根据最近K邻的数据获取出现次数最多的类别
        return y

knn = KNN()
print("开始测试KNN...")
result= knn.predict(test_data, train_data, train_label) #   使用KNN分类器对测试集中的数据进行分类测试
print("测试完成！")
print('在测试集上的准确率为: %f' % (np.mean(test_label == result))) #   计算预测的正确率
