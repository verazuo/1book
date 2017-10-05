print(__doc__)


# 加载相关库文件
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score

# 加载IRIS数据集
iris = datasets.load_iris()
X = iris.data[:, :2] # 特征
Y = iris.target # 标签

h = .02

# 使用逻辑回归训练并预测
logreg = linear_model.LogisticRegression(C=1e5) # C为正则化系数

logreg.fit(X, Y) # 训练模型

# 篡改了一下原始数据然后进行验证？还有这种操作？？？
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()]) # 预测
# 用从前的数据来验证
Z = logreg.predict(X) # 预测


## The coefficients 回归系数
print 'Coefficients: \n', logreg.coef_
## The mean squared error 均方误差
print "Mean squared error: %.2f" % mean_squared_error(Y, Z)
## 解释方差的分数: 1 is 最优预测模型
print 'Variance score: %.2f' % r2_score(Y, Z)

# 可视化结果
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)


plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()

