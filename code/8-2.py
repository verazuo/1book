# -*- coding:utf-8 -*-

import re
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation

os.getcwd()
os.chdir("C:\\Users\\verazuo\\Desktop\\lesson\\code")

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, datasets

# 加载单个文件
def load_one_flle(filename):
    x=[]
    with open(filename) as f:
        line=f.readline()
        line=line.strip('\n')
    return line

# 加载ADFA-LD中的正常样本数据
def load_adfa_training_files(rootdir):
    x=[]
    y=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            x.append(load_one_flle(path))
            print ("Load file(%s)" % path)
            y.append(0)
    return x,y

# 遍历目录下的文件
def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

# 从攻击数据集中筛选和java溢出攻击相关的数据
def load_adfa_java_files(rootdir):
    x=[]
    y=[]
    allfile=dirlist(rootdir,[])
    for file in allfile:
        if re.match(r"../data/ADFA-LD/Attack_Data_Master/Java_Meterpreter_\d*",file):
            print ("Load file(%s)" % file)
            x.append(load_one_flle(file))
            y.append(1)
    return x,y


def load_adfa_webshell_files(rootdir):
    x=[]
    y=[]
    allfile=dirlist(rootdir,[])
    for file in allfile:
        if re.match(r"../data/ADFA-LD/Attack_Data_Master/Web_Shell_\d*",file):
            print ("Load file(%s)" % file)
            x.append(load_one_flle(file))
            y.append(1)
    return x,y

if __name__ == '__main__':
    # 获取数据
    x1,y1=load_adfa_training_files("../data/ADFA-LD/Training_Data_Master/")
    # 加载
    x2,y2=load_adfa_webshell_files("../data/ADFA-LD/Attack_Data_Master/")
    
    # 用第3章中的词集模型进行特征化
    x=x1+x2
    y=y1+y2
    #print x
    
    # 实例化分词对象
    vectorizer = CountVectorizer(min_df=1)
    # 对文本进行词袋处理    
    x=vectorizer.fit_transform(x)
    # 获取对应的特征名称
    vectorizer.get_feature_names()
    # 对词袋进行向量化
    x=x.toarray()

    mlp = MLPClassifier(hidden_layer_sizes=(150,50), max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    # 训练样本
    # 创建逻辑回归模型
    logreg = linear_model.LogisticRegression(C=1e5)
    
    # 交叉验证
    score=cross_validation.cross_val_score(logreg, x, y, n_jobs=-1, cv=10)
    print  (np.mean(score))
 