from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

'''
#SGD

'''
# iris=load_iris()
# # select features and labels
# x_iris,y_iris=iris.data,iris.target
# x,y=x_iris[:,:2],y_iris
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
# x_train=preprocessing.StandardScaler().fit_transform(x_train)
# x_test=preprocessing.StandardScaler().fit_transform(x_test)
# clf=SGDClassifier()
# clf.fit(x_train,y_train)
# y_hat=clf.predict(x_test)
# y_train_hat=clf.predict(x_train)
from sklearn import metrics
# MSE=metrics.mean_squared_error(y_test,y_hat)
# acc=metrics.accuracy_score(y_test,y_hat)
# print('MSE:',MSE,'\n','acc:',acc
#
# results=metrics.classification_report(y_test,y_hat,target_names=iris.target_names)
# print(results)

'''
# KFold
'''
# from sklearn.model_selection import KFold,cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
#
# # 这里使用pipeline，便于精简模型搭建，一般而言，模型在fit之前，对数据需要feature_extraction,preprocessing,等必要步骤
# # 使用默认参数配置
# clf=Pipeline([('scaler',StandardScaler()),('sgd_clf',SGDClassifier())])
# # 5折交叉验证真个数据集合
# KF=KFold(n_splits=5, shuffle=True, random_state = 33)
# scores=cross_val_score(clf,x,y,cv=KF)
# print(scores)
# print(scores.mean(),scores.std())

'''
 SVM
'''
# from sklearn.datasets import fetch_olivetti_faces
# faces = fetch_olivetti_faces()
#
# # faces数据以Dict格式存储,与多数实验性数据格式一致
# print(faces.keys())
# print(faces.data.shape)
# print(faces.target.shape)
# from sklearn.svm import SVC
#
# x_train,x_test,y_train,y_test=train_test_split(faces.data,faces.target,test_size=0.25,random_state=0)
# from scipy.stats import sem
# # 构造一个便于交叉验证模型性能的函数(模块)
# def evaluate_cross_validation(clf,x,y,k):
#     # KFold 需要如下参数:数据量,交叉次数,是否洗牌
#     cv=KFold(k,shuffle=True,random_state=0)
#     # 采用上述方式进行交叉验证,测试模型性能,对于分类问题,默认得分accuracy
#     scores=cross_val_score(clf,x,y,cv=cv)
#     print(scores)
#     print('MEAN SCORE:%.3f(+/-%.3f)'%(scores.mean(),sem(scores)))
# # 使用线性核的SVC
# svc_linear=SVC(kernel='linear')
# # 五折交叉验证K=5
# evaluate_cross_validation(svc_linear,x_train,y_train,k=5)


'''
# # 朴素贝叶斯(Naive Bayes)
# # 大量试验证明朴素贝叶斯模型在文本分类中性能表现良好.文本特征独立性较强,刚好模型的假设便是独立特征
'''
# from sklearn.datasets import fetch_20newsgroups
# # fetch : 临时下载函数 下载20newsgroups 数据
# news=fetch_20newsgroups(subset='all')
# print(news.keys())
# print(len(news.data),len(news.target))
# print(news.target_names)
#
# # 选取25%数据用于测试模型
#
# x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)
# # 许多原始数据无法被分类器使用,图像可使用pixel信息,文本需进一步处理成数值化信息
# from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from scipy.stats import sem
# from sklearn.metrics import classification_report
# # 在naive bayes classifier的基础上,对比几种特征抽取模型的性能,并使用pipline简化构建训练流程
#
# clf_1=Pipeline([('count_vec',CountVectorizer()),('mnb',MultinomialNB())])
# clf_2=Pipeline([('Hash_vec',HashingVectorizer(non_negative=True)),('mnb',MultinomialNB())])
# clf_3=Pipeline([('Tfidf_vec',TfidfVectorizer()),('mnb',MultinomialNB())])
#
# # 构造一个便于交叉验证模型性能的模块
#
# def evaluate_cross_validation(clf,x,y,k):
#     cv=KFold(n_splits=k,shuffle=True,random_state=0)
#     scores=cross_val_score(clf,x,y,cv=cv)
#     clf.fit(x,y)
#     y_hat=clf.predict(x_test)
#     print(scores)
#     print('Mean score: %.3f (+/-%.3f)' % (scores.mean(),sem(scores)))
    # print(classification_report(y_test,y_hat))
# clfs=[clf_1,clf_2,clf_3]
# for i in clfs:
#     evaluate_cross_validation(clf=i,x=x_train,y=y_train,k=5)
# import numpy as np
# temp_alpha=np.arange(0.01,1,0.05)
# for temp in temp_alpha:
#     clf_4 = Pipeline([('Tfidf_vec_adv', TfidfVectorizer(stop_words='english')), ('mnb', MultinomialNB(alpha=temp))])
#     evaluate_cross_validation(clf_4,x_train,y_train,k=5)
# evaluate_cross_validation(clf_3,x_train,y_train,k=5)

# path='newspaper.csv'
# with open(path,'w',encoding='utf-8') as f:
# data=str(news.data)
#     f.write(data)
#
'''
决策树 Decision Tree / ensemble Tree
使用titanic 数据集实践预测乘客是否获救的分类器
'''
# import pandas as pd
# import numpy as np
# import csv
# # titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# # titanic.to_csv('titanic.csv',encoding='utf-8')
# titanic=pd.read_csv('titanic.csv',encoding='utf-8').drop(columns=['Unnamed: 0','row.names'])
# # print(titanic.head())
# # print(titanic.info())
#
# # 本模型使用pclass,age,sex进行研究
#
# x=titanic[['pclass','age','sex']]
# y=titanic['survived']
# x['age'].fillna(x['age'].mean(),inplace=True)
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
# from sklearn.feature_extraction import DictVectorizer
# vec=DictVectorizer(sparse=False)
# x_train=vec.fit_transform(x_train.to_dict(orient='record'))
# x_test=vec.transform(x_test.to_dict(orient='record'))
# # print(vec.feature_names_)
# dtc=DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=5)
# gbc=GradientBoostingClassifier(learning_rate=0.1,max_depth=3,min_samples_leaf=5)
# rfc=RandomForestClassifier(criterion='gini',max_depth=3,min_samples_leaf=5)
# model=[dtc,gbc,rfc]
# for clf in model:
#     clf.fit(x_train,y_train)
#     acc=clf.score(x_test,y_test)
#     y_hat=clf.predict(x_test)
#     mes=metrics.mean_squared_error(y_hat,y_test)
#     print(clf.__module__,'\nACC:%.3f\nMES:%.3f'%(acc,mes))


# REGRESSION 回归问题
'''
回归问题和分类问题同属于监督学习范畴
回归问题预测目标是连续实数域,预测房价,股票价格等
分类问题有限范围中离散类别进行预测.
'''

# # 波士顿房价预测问题,经典回归案例
# # load boston房价数据
#
# from sklearn.datasets import load_boston
# boston=load_boston()
# # 检查数据规模
# # print(boston.keys())
# # print(boston.data.shape)
# # print(boston.target.shape)
# # print(boston.feature_names)
# # print(boston.DESCR)
# # 检验数据是否正规化,此步骤一般情况下可省略
# import numpy as np
# print(np.max(boston.target),'\t',
#       np.min(boston.target),'\t',
#       np.mean(boston.target))
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,test_size=0.25,random_state=33)
# from sklearn.preprocessing import StandardScaler
# # print(x_train.shape)
#
# # 正规化的目的在于避免原始特征值差异过大,导致训练得到的参数权重不一
# scaler_x=StandardScaler().fit(x_train)
# x_train=scaler_x.transform(x_train)
# x_test=scaler_x.transform(x_test)
# from sklearn.model_selection import KFold,cross_val_score
# def evaluate_cross_train(clf,x_train,y_train,k):
#     cv=KFold(n_splits=k,shuffle=True,random_state=33)
#     scores=cross_val_score(clf,x_train,y_train,cv=cv)
#     print(clf.__class__)
#     print('\tAverage coefficient of determination using 5-fold cross validation:',np.mean(scores))
#     clf.fit(x_train,y_train)
#     y_hat=clf.predict(x_test)
#     sco=clf.score(x_test,y_test)
#     mse=metrics.mean_squared_error(y_test,y_hat)
#     print('\ty Predict MSE:',mse)
#     print('\ttest score:',sco)
# # 线性模型 SGD_Regressor
# from sklearn import linear_model
# SGD=linear_model.SGDRegressor(loss='squared_loss',penalty=None,random_state=33)
# evaluate_cross_train(SGD,x_train,y_train,5)
#
# # SVM Regressor
# from sklearn.svm import SVR
# svr=SVR(kernel='rbf')
# evaluate_cross_train(svr,x_train,y_train,5)
# # LinearRegression
# from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV
# lr=LinearRegression()
# lasso=LassoCV()
# ridge=RidgeCV()
# evaluate_cross_train(lr,x_train,y_train,5)
# evaluate_cross_train(lasso,x_train,y_train,5)
# evaluate_cross_train(ridge,x_train,y_train,5)
# from sklearn.ensemble import ExtraTreesRegressor    #这个随即森林回归最牛逼!!!!!!!
# etr=ExtraTreesRegressor()
# evaluate_cross_train(etr,x_train,y_train,5)
# from sklearn.ensemble import RandomForestRegressor
# rfr=RandomForestRegressor()
# evaluate_cross_train(rfr,x_train,y_train,5)



'''
无监督学习
无监督学习与监督学习区别在于没有预测/学习目标
无监督学习问题往往数据资源更加丰富,寻找数据特征自身之间的共性
监督学习两大基本类型:分类classification 和 回归 regression
无监督学习:聚类 降维
'''