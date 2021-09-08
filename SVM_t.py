import os
import cv2
import math
import time
import numpy as np
import tqdm
from skimage.feature import hog

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

    def train(self, X, y, learning_rate=1e-5, reg=0.1, delta=1, epoch=2000, batch_size=200):
        #   使用随机梯度下降法对SVM进行训练
        num_train, dim = X.shape
        y = np.array(y).astype("int")
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

class Classifier(object):
    def __init__(self, filePath):
        # 指明CIFAR10数据集所在的路径
        self.filePath = filePath
 
    def unpickle(self, fileName):
        import pickle
        # 按字典格式读取CIFAR10数据集，以字节为单位进行二进制读取
        with open(fileName, 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
        return dict
 
    def get_data(self):
        TrainData = []
        TestData = []
        # 加载训练数据集，data_batch_1~data_batch_5中各有10000条训练数据
        for i in range(1,6):
            fileName = os.path.join(self.filePath, 'data_batch_'+str(i))
            data = self.unpickle(fileName)
            train = np.reshape(data[b'data'], (10000, 3, 32 * 32))  # 图像大小32*32，RGB通道
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            # 分别读取到图像内容、图像标签和图像名称数据，一一对应起来形成三元组
            TrainData.extend(zip(train, labels, fileNames))
        # 加载测试集数据, test_batch中有10000条测试数据
        fileName = os.path.join(self.filePath,'test_batch')
        data = self.unpickle(fileName)
        test = np.reshape(data[b'data'], (10000, 3, 32 * 32))   # 图像大小32*32，RGB通道
        labels = np.reshape(data[b'labels'], (10000, 1))
        fileNames = np.reshape(data[b'filenames'], (10000, 1))
        # 分别读取到图像内容、图像标签和图像名称数据，一一对应起来形成三元组
        TestData.extend(zip(test, labels, fileNames))
        print("data read finished!")
        return TrainData, TestData
 
    def get_hog_feat(self, image, stride=8, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        # HOG特征提取过程中每个cell的规模
        cx, cy = pixels_per_cell
        # HOG特征提取过程中每个block的规模
        bx, by = cells_per_block
        sx, sy = image.shape
        # 分别计算水平和垂直方向上cell的数量
        n_cellsx = int(np.floor(sx // cx))
        n_cellsy = int(np.floor(sy // cy))
        # 分别计算水平和垂直方向上block的数量
        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        # 对于图像的每个位置，将该处的水平和垂直梯度初始值置为0
        gx = np.zeros((sx, sy), dtype=np.float32)
        gy = np.zeros((sx, sy), dtype=np.float32)
        eps = 1e-5
        # 对于图像的每个位置，需要存储梯度对应的方向和幅值，因此需要长度为2的第三维
        grad = np.zeros((sx, sy, 2), dtype=np.float32)
        for i in range(1, sx-1):
            for j in range(1, sy-1):
                # 计算水平梯度值
                gx[i, j] = image[i, j-1] - image[i, j+1]
                # 计算垂直梯度值
                gy[i, j] = image[i+1, j] - image[i-1, j]
                # 计算梯度方向（使用弧度制表示）
                grad[i, j, 0] = np.arctan(gy[i, j] / (gx[i, j] + eps)) * 180 / math.pi
                if gx[i, j] < 0:
                    grad[i, j, 0] += 180
                grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
                # 计算梯度幅值
                grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)
        # 准备求各个block的方向梯度直方图并进行归一化
        normalised_blocks = np.zeros((n_blocksy, n_blocksx, by * bx * orientations))
        for y in range(n_blocksy):
            for x in range(n_blocksx):
                # 求每一个block中的方向梯度直方图
                block = grad[y*stride:y*stride+16, x*stride:x*stride+16]    # 分离出一个block
                hist_block = np.zeros(32, dtype=np.float32) # 初始化block内方向梯度直方图为0
                eps = 1e-5
                for k in range(by):
                    for m in range(bx):
                        cell = block[k*8:(k+1)*8, m*8:(m+1)*8]  # 分离出一个cell
                        hist_cell = np.zeros(8, dtype=np.float32)   # 初始化cell内方向梯度直方图为0
                        for i in range(cy):
                            for j in range(cx):
                                n = int(cell[i, j, 0] / 45) # 计算梯度方向是8个方向中的哪一个
                                hist_cell[n] += cell[i, j, 1] # 对应方向加上梯度的幅值，即计算cell内方向梯度直方图项
                        # 将cell内的方向梯度直方图填入数组中的对应位置，即将cell的梯度直方图串联起来得到block内的方向梯度直方图  
                        hist_block[(k * bx + m) * orientations:(k * bx + m + 1) * orientations] = hist_cell[:]
                # 对每个block内的方向梯度直方图进行归一化
                normalised_blocks[y, x, :] = hist_block / np.sqrt(hist_block.sum() ** 2 + eps)
        return normalised_blocks.ravel() # 将所有block的梯度方向直方图压缩成一维数组

    def get_feat(self, TrainData, TestData):
        train_feat = []
        test_feat = []
        for data in tqdm.tqdm(TestData):
            image = np.reshape(data[0].T, (32, 32, 3))
            #   将RGB通道图像转化为灰度图，便于HOG特征提取时计算梯度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
            #   对测试集的每张图片进行HOG特征提取
            fd = self.get_hog_feat(gray)
            #   将转化为灰度图后的图像和图像标签一起存储为test_feat的一个元素，舍弃图像名称
            fd = np.concatenate((fd, data[1]))
            test_feat.append(fd)
        #   将测试集HOG特征转为float型数组
        test_feat = np.array(test_feat).astype("float")
        #   将测试集HOG特征信息存储到文件中，便于后续使用
        np.save("test_feat.npy", test_feat)
        print("Test features are extracted and saved.")
        for data in tqdm.tqdm(TrainData):
            image = np.reshape(data[0].T, (32, 32, 3))
            #   将RGB通道图像转化为灰度图，便于HOG特征提取时计算梯度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            #   对训练集的每张图片进行HOG特征提取
            fd = self.get_hog_feat(gray)
            #   将转化为灰度图后的图像和图像标签一起存储为test_feat的一个元素，舍弃图像名称
            fd = np.concatenate((fd, data[1]))
            train_feat.append(fd)
        #   将训练集HOG特征转为float型数组
        train_feat = np.array(train_feat).astype("float")
        #   将训练集HOG特征信息存储到文件中，便于后续使用
        np.save("train_feat.npy", train_feat)
        print("Train features are extracted and saved.")
        return train_feat, test_feat
 
    def classification(self, train_feat, test_feat):
        # 记录训练和预测的起始时间
        t0 = time.time()
        # 建立SVM分类器实例
        clf = SVM()
        print("Training a Linear SVM Classifier.")
        # 用训练集图像提取出的HOG特征训练SVM分类器
        clf.train(train_feat[:,:-1], train_feat[:,-1], learning_rate=1e-7, reg=5e4)
        # 对测试集图像进行预测
        predict_result = clf.predict(test_feat[:,:-1])
        num = 0
        for i in range(len(predict_result)):
            if int(predict_result[i]) == int(test_feat[i,-1]):
                num += 1
        # 计算测试集预测正确率
        rate = float(num) / len(predict_result)
        # 记录训练和预测的完成时间
        t1 = time.time()
        print('The testing classification accuracy is %f' % rate)
        print('The testing cost of time is :%f' % (t1 - t0))
        # 对训练集图像进行预测
        predict_result2 = clf.predict(train_feat[:,:-1])
        num2 = 0
        for i in range(len(predict_result2)):
            if int(predict_result2[i]) == int(train_feat[i,-1]):
                num2 += 1
        # 计算训练集预测正确率
        rate2 = float(num2) / len(predict_result2)
        print('The Training classification accuracy is %f' % rate2)

    def run(self):
        if os.path.exists("train_feat.npy") and os.path.exists("test_feat.npy"):
            # 已经存在HOG特征提取的结果，则直接使用即可。避免再次HOG提取占用大量时间
            train_feat = np.load("train_feat.npy")
            test_feat = np.load("test_feat.npy")
        else:
            # 不存在HOG特征提取的结果，则直接需要首先加载原数据，再对原数据做HOG特征提取获得梯度特征，最终是使用每张图片的梯度特征训练SVM并进行预测
            train_data, test_data = self.get_data()
            train_feat, test_feat = self.get_feat(train_data, test_data)
        # 数据预处理，在图像分类任务的实践中，对每个特征减去平均值来中心化数据也是非常重要的
        train_feat = train_feat - np.mean(train_feat, axis=0)
        test_feat = test_feat - np.mean(test_feat, axis=0)
        # 对训练集和测试集的每条数据，额外增加一个数值为常量1的维度，用以将W和b两个超参数简化为W一个超参数，即偏差和权重合并
        train_feat = np.hstack((train_feat, np.ones([train_feat.shape[0], 1]))) 
        test_feat = np.hstack((test_feat, np.ones([test_feat.shape[0], 1])))
        self.classification(train_feat, test_feat)
 
if __name__ == '__main__':
    filePath = r'datasets'
    cf = Classifier(filePath)
    cf.run()
