#coding:utf-8

def hehe():
	print "hehe"


def Test01():
	from sklearn import datasets
	iris = datasets.load_iris()    #load data
	digits = datasets.load_digits()
	print digits.images[0]  #老子是中文注释！！

def Test02():
	from sklearn import datasets
	iris = datasets.load_iris() #这里花的数据
	data = iris.data
	#print data.shape
	#print iris.DESCR
	digits = datasets.load_digits() #这是数字的数据
	#print digits.images.shape

	import pylab as pl
	pl.imshow(digits.images[-1], cmap = pl.cm.gray_r)
	#pl.show()

	data = digits.images.reshape(digits.images.shape[0], -1); #这里的data是一个数组，每一行是64维特征

	print len(data)
	print len(data[-1])

def supervisedTest01():
	import numpy as np
	from sklearn import datasets
	iris = datasets.load_iris()
	iris_X = iris.data   #iris_X 是150*4的特征（二维矩阵）
	iris_Y = iris.target #iris_Y 是150*1的label (1维向量)

	#print len(iris_X)
	#print len(iris_Y)
	#print np.unique(iris_Y) #这是获得label的种类， 这里是一共3类

	np.random.seed(0)
	indices = np.random.permutation(len(iris_X)) #获得0-149的一个全排列
	#print indices

	iris_x_train = iris_X[indices[:-10]] #这里是获得从开始到倒数第十的数据
	iris_y_train = iris_Y[indices[:-10]] #获得与iris_x_train对应的label
	iris_x_test  = iris_X[indices[-10:]] #这是获得最后的10组数据作为test数据
	iris_y_test  = iris_Y[indices[-10:]] ##获得与iris_x_test对应的label

	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_neighbors=3, p=2, weights='uniform')
	knn.fit(iris_x_train, iris_y_train) #其实这里输入的就是train_x 和train_y
										#算法待理解
	print knn.predict(iris_x_test)
	print iris_y_test


def supervisedTest02():
	import numpy as np
	from sklearn import datasets

	diabetes = datasets.load_diabetes()
	diabetes_X_train = diabetes.data[:-20]
	diabetes_X_test  = diabetes.data[-20:]
	diabetes_Y_train = diabetes.target[:-20]
	diabetes_Y_test  = diabetes.target[-20:]

	from sklearn import linear_model
	regr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
	regr.fit(diabetes_X_train, diabetes_Y_train)

	#print regr.coef_  #注意因为diabetes_X_train的特征是4维，所以coef_的个数是4+1 = 5

	mean_err = np.mean((regr.predict(diabetes_X_test) - diabetes_Y_test) ** 2)
	score = regr.score(diabetes_X_test, diabetes_Y_test) #这是判断test数据预测程度

	print mean_err
	print score


	print len(diabetes.data)    #样本数目
	print len(diabetes.data[0]) #特征维数


def supervisedTest03():
	import numpy as np
	from sklearn import linear_model

	from sklearn import datasets

	diabetes = datasets.load_diabetes()
	diabetes_X_train = diabetes.data[:-20]
	diabetes_X_test  = diabetes.data[-20:]
	diabetes_Y_train = diabetes.target[:-20]
	diabetes_Y_test  = diabetes.target[-20:]

	X = np.c_[ .5, 1].T
	y = [0.5, 1]
	test = np.c_[0, 2].T

	regr = linear_model.LinearRegression() #这里用的是LinearRegression

	import pylab as pl
	pl.figure(1)

	np.random.seed(0)
	for _ in range(6):
		this_X = .1 * np.random.normal(size=(2,1)) + X
		regr.fit(this_X, y)
		pl.plot(test, regr.predict(test))
		pl.scatter(this_X, y, s=3)


	#pl.show()
	regr = linear_model.Ridge(alpha=.1) #这里用的是Ridge, 都是linear_model

	pl.figure(4)
	np.random.seed(0)
	for _ in range(6):
		this_X = .1 * np.random.normal(size=(2,1)) + X
		regr.fit(this_X, y)  # 这里是调用了estimator的fit函数
		pl.plot(test, regr.predict(test))
		pl.scatter(this_X, y, s=3)

	pl.show()

	#alphas = np.logspace(-4, -1, 6) #从-4到-1， 分6份
	#from __future__ import print_function
	#
	#aaa = [regr.set_params(alpha = alpha).fit(diabetes_X_train, diabetes_Y_train).score(diabetes_X_test, diabetes_Y_test) for alpha in alphas]






def plotTest01():
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from sklearn import datasets
	from sklearn.decomposition import PCA

	iris = datasets.load_iris()
	X = iris.data[:,:2] # only first two features
	Y = iris.target

	x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
	y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

	plt.figure(2, figsize = (8, 6))
	plt.clf()

	plt.scatter(X[:, 0], X[:, 1], c = Y, cmap = plt.cm.Paired)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())

	#plt.show()

	fig = plt.figure(1, figsize = (8, 6))
	ax = Axes3D(fig, elev = -150, azim = 110)
	X_reduced = PCA(n_components = 3).fit_transform(iris.data)
	#这里是画散点图
	ax.scatter(X_reduced[:,0], X_reduced[:, 1], X_reduced[:,2], c = Y, cmap = plt.cm.Paired)
	ax.set_title('First three PCA directionsa')
	ax.set_xlabel('1st eigenvector')
	ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel('2nd eigenvector')
	ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel('3rd eigenvector')
	ax.w_zaxis.set_ticklabels([])

	plt.show()



def DigitClassificationTest01():
	from sklearn import datasets, neighbors, linear_model
	import numpy as np

	digits = datasets.load_digits()
	X_digits = digits.data
	Y_digits = digits.target

	n_samples = len(X_digits)

	X_train = X_digits[:.9 * n_samples]
	Y_train = Y_digits[:.9 * n_samples]
	X_test = X_digits[.9 * n_samples:]
	Y_test = Y_digits[.9 * n_samples:]

	knn = neighbors.KNeighborsClassifier()
	logistic = linear_model.LogisticRegression()

	knn_score = knn.fit(X_train, Y_train).score(X_test, Y_test)
	logistic_score = logistic.fit(X_train, Y_train).score(X_test, Y_test)


	#print knn_score
	#print logistic_score
	#这个准确程度可以到90%+

	#print len(X_train) #X_train是1617 * 64维的矩阵

	#print Y_train[1] #这里的y就是对应的0-9, 这是一个多类分类的问题，训练的时候也是直接用的fit函数


	iris = datasets.load_iris()
	iris_X = iris.data   #iris_X 是150*4的特征（二维矩阵）
	iris_Y = iris.target #iris_Y 是150*1的label (1维向量)

	np.random.seed(0)
	indices = np.random.permutation(len(iris_X)) #获得0-149的一个全排列

	iris_x_train = iris_X[indices[:-50]]
	iris_y_train = iris_Y[indices[:-50]]
	iris_x_test  = iris_X[indices[-50:]]
	iris_y_test  = iris_Y[indices[-50:]]

	from sklearn import svm

	#svc = svm.SVC(kernel = 'linear')  # SVR -- Support Vector Regression, # SVC -- Support Vector Classification
	svc = svm.SVC(kernel = 'poly', degree = 3)
	#svc = svm.SVC(kernel = 'rbf')
	svc.fit(iris_x_train, iris_y_train)

	score = svc.score(iris_x_test, iris_y_test)

	print score



def ModelSelectionTest01():
	from sklearn import datasets, svm
	import numpy as np
	digits = datasets.load_digits()
	X_digits = digits.data
	Y_digits = digits.target
	svc = svm.SVC(C = 1, kernel = 'linear')
	score = svc.fit(X_digits[:-100], Y_digits[:-100]).score(X_digits[-100:], Y_digits[-100:])

	#print score

	X_folds = np.array_split(X_digits, 3)
	Y_folds = np.array_split(Y_digits, 3)

	#print len(X_folds[0])

	scores = list()

	for k in range(3):
		X_train = list(X_folds) #这里的X_folds是一个具有3个元素的list
		X_test = X_train.pop(k) #test是train的第K个元素
		X_train = np.concatenate(X_train) #这里是把X_train减去X_test
		#print len(X_train)
		Y_train = list(Y_folds)
		Y_test = Y_train.pop(k)
		Y_train = np.concatenate(Y_train)

		scores.append(svc.fit(X_train, Y_train).score(X_test, Y_test))

	#print scores


	from sklearn import cross_validation
	k_fold = cross_validation.KFold(n = 6, n_folds = 3)
	for train_indices, test_indices in k_fold:
		print train_indices, test_indices

	k_fold = cross_validation.KFold(len(X_digits), n_folds = 3)
	scores = [svc.fit(X_digits[train], Y_digits[train]).score(X_digits[test], Y_digits[test]) for train , test in k_fold]

	#print scores

	scores = cross_validation.cross_val_score(svc, X_digits, Y_digits, cv = k_fold, n_jobs = 1)
	#print scores

	from sklearn.grid_search import GridSearchCV
	gammas = np.logspace(-6, -1, 10)
	clf = GridSearchCV(estimator = svc, param_grid = dict(gamma = gammas), n_jobs = 1)
	clf.fit(X_digits[:1000], Y_digits[:1000])
	print clf.best_score_
	print clf.best_estimator_.gamma

	from sklearn import linear_model, datasets
	lasso = linear_model.LassoCV()    #这里的lassoCV和lasso有什么区别？
	diabetes = datasets.load_diabetes()
	X_diabetes = diabetes.data
	Y_diabetes = diabetes.target
	lasso.fit(X_diabetes, Y_diabetes)

	print lasso.alpha_



def unsupervisedLearningTest01():
	from sklearn import cluster, datasets
	iris = datasets.load_iris()
	X_iris = iris.data
	Y_iris = iris.target

	k_means = cluster.KMeans(n_clusters = 3) #这里是设置k-means的中心数
	k_means.fit(X_iris)						# fit data

	print k_means.labels_[::10]


def unsupervisedLearningTest02():
	from sklearn import cluster
	import scipy as sp
	import numpy as np
	try:
		lena = sp.lena()
	except AttributeError:
		from scipy import misc
		lena = misc.lena()

	X = lena.reshape((-1, 1))
	k_means = cluster.KMeans(n_clusters = 5, n_init = 1)
	k_means.fit(X)
	values = k_means.cluster_centers_.squeeze()
	labels = k_means.labels_
	lena_compressed = np.choose(labels, values)
	lena_compressed.shape = lena.shape

	print lena_compressed



def plotTest02():
	import numpy as np
	import scipy as sp
	import matplotlib.pyplot as plt

	from sklearn import cluster

	n_clusters = 5
	np.random.seed(0)

	try:
		lena = sp.lena()
	except AttributeError:
		from scipy import misc
		lena = misc.lena()

	X = lena.reshape((-1, 1))
	k_means = cluster.KMeans(n_clusters = n_clusters, n_init = 4)
	k_means.fit(X)

	values = k_means.cluster_centers_.squeeze() #这是获得聚类中心
	labels = k_means.labels_

	lena_compressed = np.choose(labels, values)
	lena_compressed.shape = lena.shape

	vmin = lena.min()
	vmax = lena.max()

	#original
	plt.figure(1, figsize = (3, 2.2))
	plt.imshow(lena, cmap = plt.cm.gray, vmin = vmin, vmax = vmax)


	#compressed data
	plt.figure(2, figsize = (3, 2.2))
	plt.imshow(lena_compressed, cmap = plt.cm.gray, vmin = vmin, vmax = vmax)

	#这里面有一些函数要搞清楚意义是什么
	#equal bins lena
	regular_values = np.linspace(0, 256, n_clusters + 1)
	regular_labels = np.searchsorted(regular_values, lena) - 1
	regular_values = 0.5 * (regular_values[1:] + regular_values[:-1]) #mean
	regular_lena = np.choose(regular_labels.ravel(), regular_values)

	regular_lena.shape = lena.shape

	plt.figure(3, figsize=(3, 2.2))
	plt.imshow(regular_lena, cmap = plt.cm.gray, vmin = vmin, vmax = vmax)


	#histogram
	plt.figure(4, figsize = (3, 2.2))
	plt.clf()
	plt.axes([0.01, 0.01, 0.98, 0.98])
	plt.hist(X, bins = 256, color = '0.5', edgecolor = '.5')
	plt.yticks()
	plt.xticks(regular_values)

	values = np.sort(values)
	for center_1, center_2 in zip(values[:-1], values[1:]):
		plt.axvline(0.5 * (center_1 + center_2), color = 'b')

	for center_1, center_1 in zip(regular_values[:-1], regular_values[1:]):
		plt.axvline(0.5 * (center_1 + center_2), color = 'b', linestyle = '--')

	plt.show()



def unsupervisedLearningTest03():
	# Connectivity-constrained clustering

	import numpy as np
	import scipy as sp
	import matplotlib.pyplot as plt
	import time

	from sklearn.feature_extraction.image import grid_to_graph
	from sklearn.cluster import AgglomerativeClustering
	from sklearn import cluster, datasets
	lena = sp.misc.lena()

	#Downsample the image by a factor of 4
	lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]

	X = np.reshape(lena, (-1, 1))

	# Define the structure A of the data. Pixels connected to their neighbors.
	# 把图片变成一张图， 讨论其连接性
	connectivity = grid_to_graph(*lena.shape)


	print "Compute structured hierarchical clustering..."
	st = time.time()

	n_clusters = 15 # number of regions

	ward = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward', connectivity = connectivity).fit(X)
	label = np.reshape(ward.labels_, lena.shape)
	print "Elapsed time: " + str(time.time() - st)
	print "Number of pixels: " + str(label.size)
	print "Number of clusters: " + str(np.unique(label).size)

	#Feature agglomeration
	digits = datasets.load_digits()
	images = digits.images
	X = np.reshape(images, (len(images), -1))

	connectivity = grid_to_graph(*images[0].shape)

	agglo = cluster.FeatureAgglomeration(connectivity = connectivity, n_clusters = 32)

	agglo.fit(X)
	X_reduced = agglo.transform(X)
	X_approx = agglo.inverse_transform(X_reduced)
	images_approx = np.reshape(X_approx, images.shape)




def unsupervisedLearningTest04():
	#主成分分析
	import numpy as np

	x1 = np.random.normal(size = 100)
	x2 = np.random.normal(size = 100)
	x3 = x1 + x2

	X = np.c_[x1, x2, x3]

	from sklearn import decomposition

	pca = decomposition.PCA()
	pca.fit(X)

	print pca.explained_variance_

	pca.n_components = 2
	X_reduced = pca.fit_transform(X)

	print X_reduced.shape



	#独立成分分析
	time = np.linspace(0, 10, 2000)

	s1 = np.sin(2 * time)
	s2 = np.sign(np.sin(3 * time))
	S = np.c_[s1, s2]
	S += 0.2 * np.random.normal(size = S.shape) #注意这里的shape
	S /= S.std(axis = 0)

	A = np.array([[1, 1], [0.5, 2]])
	X = np.dot(S, A.T)


	ica = decomposition.FastICA()
	S_ = ica.fit_transform(X)
	A_ = ica.mixing_.T
	np.allclose(X, np.dot(S_, A_) + ica.mean_)



def CombinationTest01():

	from sklearn import linear_model, decomposition, datasets
	from sklearn.pipeline import Pipeline
	from sklearn.grid_search import GridSearchCV
	import matplotlib.pyplot as plt
	import numpy as np

	logistic = linear_model.LogisticRegression()

	pca = decomposition.PCA()
	pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

	digits = datasets.load_digits()
	X_digits = digits.data
	Y_digits = digits.target

	pca.fit(X_digits)

	plt.figure(1, figsize = (4, 3))
	plt.clf()
	plt.axes([0.2, 0.2, 0.7, 0.7])

	plt.plot(pca.explained_variance_, linewidth = 2)
	plt.axis('tight')

	plt.xlabel('n_components')
	plt.ylabel('explained_variance_')


	#print len(pca.explained_variance_) #pca.explained_variance_ 是64*1的向量(和特征维数相同)，从大到小

	#plt.show()

	#Prediction
	n_components = [20, 40, 64]

	Cs = np.logspace(-4, 4, 3)

	estimator = GridSearchCV(pipe, dict(pca__n_components = n_components, logistic__C = Cs))
	estimator.fit(X_digits, Y_digits)

	plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components, linestyle = ':', label = 'n_components chosen')
	plt.legend(prop = dict(size = 12))

	plt.show()

def lfwTest01():
	from sklearn.datasets import fetch_lfw_people
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

	for name in lfw_people.target_names:
		print(name)

def lfwTest02():
	#from __future__ import print_function
	#学习对应模块间的接口， 数据格式

	from time import time
	import logging
	import matplotlib.pyplot as plt

	from sklearn.cross_validation import train_test_split
	from sklearn.datasets import fetch_lfw_people
	from sklearn.grid_search import GridSearchCV
	from sklearn.metrics import classification_report
	from sklearn.metrics import confusion_matrix
	from sklearn.decomposition import RandomizedPCA
	from sklearn.svm import SVC

	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4) #这里的min_faces_per_person是用来限制读取图片的数据
	#lfw_people = fetch_lfw_people(min_faces_per_person=5, resize=0.4)

	# introspect the images arrays to find the shapes (for plotting)
	n_samples, h, w = lfw_people.images.shape

	X = lfw_people.data
	n_features = X.shape[1]

	# the label to predict is the id of the person
	y = lfw_people.target
	target_names = lfw_people.target_names
	n_classes = target_names.shape[0]

	#print target_names.shape

	#print("Total dataset size:")
	#print("n_samples: %d" % n_samples)
	#print("n_features: %d" % n_features)
	#print("n_classes: %d" % n_classes)
	#print("h: %d" % h)
	#print("w: %d" % w)
	#print lfw_people
	#print target_names


	###############################################################################
	# Split into a training set and a test set using a stratified k fold

	# split into a training and testing set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

	#X_train 是966 * 1850的二维矩阵

	print X_train
	print len(X_train)
	print len(X_train[0])

	#singleImage = X[1].reshape(h, w)
	#
	##显示图片
	#plt.imshow(singleImage, cmap = plt.cm.gray_r)
	#plt.show()


	# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
	# dataset): unsupervised feature extraction / dimensionality reduction
	n_components = 150

	print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
	t0 = time()
	pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train) #这里是利用PCA获得主成分
	print("done in %0.3fs" % (time() - t0))


	print pca.components_   #这里的pca.components_是150 * 1850的矩阵 -- 可以理解为就是找了150个向量， 每一向量都是原来所有样本的某一种线性组合（因此特征维数不会变）

	print len(pca.components_)
	print len(pca.components_[0])


	eigenfaces = pca.components_.reshape((n_components, h, w)) #将向量转化为图片
																#这里可以认为特征脸是原始若干张人脸的线性叠加


	print("Projecting the input data on the eigenfaces orthonormal basis")
	t0 = time()
	X_train_pca = pca.transform(X_train) #把原始图片投影到eigenfaces空间里
	X_test_pca = pca.transform(X_test)
	print("done in %0.3fs" % (time() - t0))


	#print len(eigenfaces[0])

	print X_train_pca			#X_train_pca是966 * 150维矩阵， 其150维的每一维都是原始的train向量在对应eigenface上的投影
	print len(X_train_pca)		#所以train_pca是一组float, 不是int
	print len(X_train_pca[0])

	#print y_train #y_train就是label, 0-6 表示类别

	#显示图片
	#plt.imshow(eigenfaces[-1], cmap = plt.cm.gray_r)
	#plt.show()


	# Train a SVM classification model

	print("Fitting the classifier to the training set")
	t0 = time()
	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
				  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid) #注意这里是用svm进行分类，所以是svc
	clf = clf.fit(X_train_pca, y_train) #输入为转换后的pca数据和y_train进行训练 -- 多类别svm
	print("done in %0.3fs" % (time() - t0))
	print("Best estimator found by grid search:")
	print(clf.best_estimator_)

	# Quantitative evaluation of the model quality on the test set

	print("Predicting people's names on the test set")
	t0 = time()
	y_pred = clf.predict(X_test_pca) #测试数据也是在相同的eigenface空间内进行投影后的结果
	print("done in %0.3fs" % (time() - t0))

	print(classification_report(y_test, y_pred, target_names=target_names))
	print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


	###############################################################################
	# Qualitative evaluation of the predictions using matplotlib

	def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
		"""Helper function to plot a gallery of portraits"""
		plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
		plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
		for i in range(n_row * n_col):
			plt.subplot(n_row, n_col, i + 1)
			plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
			plt.title(titles[i], size=12)
			plt.xticks(())
			plt.yticks(())


	# plot the result of the prediction on a portion of the test set

	def title(y_pred, y_test, target_names, i):
		pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
		true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
		return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

	prediction_titles = [title(y_pred, y_test, target_names, i)
						 for i in range(y_pred.shape[0])]

	plot_gallery(X_test, prediction_titles, h, w)

	# plot the gallery of the most significative eigenfaces

	eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
	plot_gallery(eigenfaces, eigenface_titles, h, w)

	plt.show()



def face_completion_Test01():
	import numpy as np
	import matplotlib.pyplot as plt

	from sklearn.datasets import fetch_olivetti_faces
	from sklearn.utils.validation import check_random_state

	from sklearn.ensemble import ExtraTreesRegressor
	from sklearn.neighbors import KNeighborsRegressor
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model import RidgeCV



	#load the faces datasets
	data = fetch_olivetti_faces()
	targets  = data.target

	#print len(data.data)
	#print len(data.data[0])  #data.data 是 400 * 4096 的数据

	#感觉这里的4096维和原图不一样啊...ravelled image
	#face = data.data[1].reshape(64,64)  #注意这里的data和image
	#face = data.images[1]
	#face_ccw_90 = zip(*face)[::-1]
	#face_cw_90 = zip(*face[::-1])

	#plt.imshow(face_cw_90, cmap = plt.cm.gray_r)
	#plt.show()

	#这里是为了做左右预测， 所以把原图旋转了90度
	#for i in range(len(data.images)):
	#	face = data.images[i]
	#	data.images[i] = face_cw_90 = zip(*face[::-1])




	#print data.images[0]
	data = data.images.reshape((len(data.images), -1)) #相当于就是data.data...把一张图片变成了一个行向量
	#print len(data[0])


	train = data[targets < 30]
	test = data[targets >= 30] #注意这里的test和targe没有关系

	n_faces = 5
	rng = check_random_state(4)

	#test.shape = [100, 4096]
	face_ids = rng.randint(test.shape[0], size = (n_faces, )) #这里相当于是在0-99中随机选择出5个数
	test = test[face_ids, :]

	#print face_ids

	n_pixels = data.shape[1]
	X_train = train[:, :np.ceil(0.5 * n_pixels)] #脸的上半部分
	Y_train = train[:, np.floor(0.5 * n_pixels):] #脸的下半部分
	X_test = test[:, :np.ceil(0.5 * n_pixels)] #相当于是那脸的前半部分预测后半部分 -- 是一个多对多的学习过程， train和test的维度相同
	Y_test = test[:, np.floor(0.5 * n_pixels):]

	#注意因为是要做completion, 所以是regression 而不是 classification
	#这里的ESTMATORS是一个字典
	ESTIMATORS = {
		"Extra trees": ExtraTreesRegressor(n_estimators = 10, max_features = 32, random_state = 0),
		"k-nn": KNeighborsRegressor(),
		"Linear regression": LinearRegression(),
		"Ridge": RidgeCV(),
	}

	#这里是直接进行预测， 也就是fit + predict的过程
	print "start fiting and predicting"
	y_test_predict = dict()
	for name, estimator in ESTIMATORS.items():
		estimator.fit(X_train, Y_train)
		y_test_predict[name] = estimator.predict(X_test)

	print "start plotting"


	#下面是画图

	image_shape = (64, 64)

	n_cols = 1 + len(ESTIMATORS)
	plt.figure(figsize=(2.0 * n_cols, 2.26 * n_faces))
	plt.suptitle("Face completion with multi-output estimators GoGoGo", size = 16)

	for i in range(n_faces):
		true_face = np.hstack((X_test[i], Y_test[i]))

		if i:
			sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
		else:
			sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title = "true faces")

		sub.axis("off")

		sub.imshow(true_face.reshape(image_shape), cmap = plt.cm.gray, interpolation = "nearest")

		#a = true_face.reshape(image_shape)
		#sub.imshow(zip(*a)[::-1], cmap = plt.cm.gray, interpolation = "nearest")


		for j, est in enumerate(sorted(ESTIMATORS)):
			completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

			if i:
				sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
			else:
				sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title = est)

			sub.axis("off")
			sub.imshow(completed_face.reshape(image_shape), cmap = plt.cm.gray, interpolation = "nearest")

			#b = completed_face.reshape(image_shape)
			#sub.imshow(zip(*b)[::-1], cmap = plt.cm.gray, interpolation = "nearest")

	plt.show()


def rotateTest():
	m = [[1,2,3],[4,5,6],[7,8,9]]
	print zip(*m)[::-1] # counter clock-wise
	print zip(*m[::-1]) # clock-wise


def OnlineLearningTest01():
	import time

	import matplotlib.pyplot as plt
	import numpy as np

	from sklearn import datasets
	from sklearn.cluster import MiniBatchKMeans
	from sklearn.feature_extraction.image import extract_patches_2d

	faces = datasets.fetch_olivetti_faces()

	print "Learning the dictionary..."
	rng = np.random.RandomState(0)

	kmeans = MiniBatchKMeans(n_clusters = 81, random_state = rng, verbose = True)
	patch_size = (20, 20)

	buffer = []
	index = 1
	t0 = time.time()

	#Online Learning
	index = 0

	for _ in range(6):
		for img in faces.images:
			data = extract_patches_2d(img, patch_size, max_patches = 50, random_state = rng)
			data = np.reshape(data, (len(data), -1))

			buffer.append(data)
			index += 1
			if index % 10 == 0:
				data = np.concatenate(buffer, axis = 0) #这里是把一个数组合并成矩阵

				#这里要先做标准化
				data -= np.mean(data, axis = 0)
				data /= np.std(data, axis = 0)
				kmeans.partial_fit(data) 	#每次都是调用partial_fit函数进行学习
				buffer = []

			if index % 100 == 0:
				print "Partial fit of %4i out of %i" % (index, 6 * len(faces.images))


	dt = time.time() - t0
	print "done in %.2fs. " % dt

	#plot result
	plt.figure(figsize = (4.2, 4))
	for i, patch in enumerate(kmeans.cluster_centers_):
		plt.subplot(9,9, i + 1)
		plt.imshow(patch.reshape(patch_size), cmap = plt.cm.gray, interpolation = "nearest")

		plt.xticks(())
		plt.xticks(())


	plt.suptitle('Patches of faces\nTrain time %.1fs on %d patches' % (dt, 8 * len(faces.images)), fontsize = 16)
	plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

	plt.show()


def Test03():
	import numpy as np

	a = [3, 4]
	b = [1, 2]
	data = []
	data.append(a)
	data.append(b)

	print data
	print np.concatenate(data, axis = 0)



	a = np.array([[ 1],[ 2],[ 3]])
	b = np.array([[ 2],[ 3],[ 4]])
	print a
	print b
	print np.hstack((a,b,a))
	print np.vstack((a,b,a))


def RBMtest01():
	#利用RBM进行non-linear feature extraction
	#相对于直接进行logistic regression， RBM features 可以提高分类精度

	import numpy as np
	import matplotlib.pyplot as plt

	from scipy.ndimage import convolve
	from sklearn import linear_model, datasets, metrics
	from sklearn.cross_validation import train_test_split
	from sklearn.neural_network import BernoulliRBM
	from sklearn.pipeline import Pipeline

	def nudge_dataset(X, Y):
		direction_vectors = [
			[[0, 1, 0],
			 [0, 0, 0],
			 [0, 0, 0]],

			[[0, 0, 0],
			 [1, 0, 0],
			 [0, 0, 0]],

			[[0, 0, 0],
			 [0, 0, 1],
			 [0, 0, 0]],

			[[0, 0, 0],
			 [0, 0, 0],
			 [0, 1, 0]]
		]

		shift = lambda x, w: convolve(x.reshape((8, 8)), mode = 'constant', weights = w).ravel()

		X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
		Y = np.concatenate([Y for _ in range(5)], axis = 0)

		return X, Y

	digits = datasets.load_digits()
	X = np.asarray(digits.data, 'float32')  #这里应该就是进行了一下数据类型转换 a#list to array

	X, Y = nudge_dataset(X, digits.target)  #相当于重新生成了5倍的X,Y

	#print np.max(X, 0)
	#print np.min(X, 0)
	X = (X - np.min(X, 0)) / (np.max(X, 0) - - np.min(X, 0) + 0.0001) # 0-1 scaling 这里做了归一化(每一维分别归一化)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


	print set(Y_train)
	#'''
	#新建模型
	logistic = linear_model.LogisticRegression()
	rbm = BernoulliRBM(random_state = 0, verbose = True)

	#感觉这里的pipeline就是一个连续进行fit, transform的过程
	#而rbm模型transform的结果是Latent representations of the data.

	classifier = Pipeline(steps = [('rbm', rbm), ('logistic', logistic)])

	#Training
	#这里的参数是根据cross-validation选出来的 -- GridSearchCV
	rbm.learning_rate = 0.06
	rbm.n_iter = 20
	rbm.n_components = 100  #这里就是利用rbm 训练出100个特征
	logistic.C = 6000


	#rbm.fit(X_train, Y_train)
	rbm.fit(X_train)


	#rbm从数据的维数来看，首先是一个非监督的训练过程，就是从X_train中求出N个代表性的vector,
	#然后再把原始的X_trian投影到这N的向量上，获得X_train的新N维feature
	#与PCA类似

	predicted_Y = rbm.transform(X_train)

	print rbm.components_  #rbm.components_是 100 * 64的矩阵
	print len(rbm.components_)
	print len(rbm.components_[0])

	print predicted_Y
	print len(predicted_Y)
	print len(predicted_Y[0])
	print len(X_train)
	print len(X_train[0])


	# Training RBM-Logistic Pipeline
	#相当于这里输入的还是每一维都进行了归一化之后的X_train
	#对应的Y_train还是0-9 表示label
	print "Start Training RBM-Logistic Pipeline"
	classifier.fit(X_train, Y_train)





	# Training Logistic regression，
	logistic_classifier = linear_model.LogisticRegression(C = 100.0)
	logistic_classifier.fit(X_train, Y_train)

	#Evaluation

	print "Logistic regression using RBM features: \n%s\n" %(metrics.classification_report(Y_test, classifier.predict(X_test)))
	print "Logistic regression using raw features: \n%s\n" %(metrics.classification_report(Y_test, logistic_classifier.predict(X_test)))


	#Plotting

	plt.figure(figsize = (4.2, 4))

	for i, comp in enumerate(rbm.components_):
		plt.subplot(10, 10, i + 1)
		#这里获得的还是100个64维vector，然后把每一个vector都reshape到8*8显示出来
		plt.imshow(comp.reshape(8,8), cmap=plt.cm.gray_r)
		plt.xticks(())
		plt.yticks(())

	plt.suptitle('100 components extracted by RBM', fontsize = 16)
	plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.23)

	plt.show()


def compressiveSensingTest01():
	import numpy as np
	from scipy import sparse
	from scipy import ndimage
	from sklearn.linear_model import Lasso
	from sklearn.linear_model import Ridge
	import matplotlib.pyplot as plt

	def _weights(x, dx=1, orig=0):
		x = np.ravel(x)
		floor_x = np.floor((x - orig) / dx)
		alpha = (x - orig - floor_x * dx) / dx
		return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))

	def _generate_center_coordinates(l_x):
		l_x = float(l_x)
		X, Y = np.mgrid[:l_x, :l_x]
		center = l_x / 2
		X += 0.5 - center
		Y += 0.5 - center
		return X, Y


	def build_projection_operator(l_x, n_dir):
		X, Y = _generate_center_coordinates(l_x)
		angles = np.linspace(0, np.pi, n_dir, endpoint = False) #这里是把0-pi平均分成了n_dir份
		data_inds, weights, camera_inds = [], [], []
		data_unravel_indices = np.arange(l_x ** 2)
		data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))

		for i, angle in enumerate(angles):
			Xrot = np.cos(angle) * X - np.sin(angle) * Y
			inds, w = _weights(Xrot, dx = 1, orig = X.min())
			mask = np.logical_and(inds >=0, inds < l_x)
			weights += list(w[mask])
			camera_inds += list(inds[mask] + i * l_x)
			data_inds += list(data_unravel_indices[mask])

		#生成稀疏矩阵
		proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))

		return proj_operator

	def generate_synthetic_data():
		#这里相当于就是生成了一张图片
		rs = np.random.RandomState(0)
		n_pts = 36.0

		x, y = np.ogrid[0:l, 0:l]
		mask_outer = (x - l / 2) ** 2 + (y - l / 2) ** 2 < ( l / 2) **2
		mask = np.zeros((l, l))
		points = l * rs.rand(2, n_pts)
		mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
		mask = ndimage.gaussian_filter(mask, sigma = l / n_pts)
		res = np.logical_and(mask > mask.mean(), mask_outer)
		return res - ndimage.binary_erosion(res)


	#generate synthetic images, and projections

	l = 128
	proj_operator = build_projection_operator(l, l / 7.0)
	data = generate_synthetic_data()
	proj = proj_operator * data.ravel()[:, np.newaxis] #[2304, 16384] * [16384, 1]

	#这里是加上一个随机数
	proj += 0.15 * np.random.randn(*proj.shape)

	#proj_operator 是一个稀疏矩阵， 共有2304行，每行是一个16384维的向量，代表了图像上某一个亮点的位置，其值表示亮度的大小
	#注意这里是直接用1维向量来表示点在图片上的位置， 而不是二维矩阵
	#学习这里的proj_opnerator是怎么生成的
	#print proj_operator.shape

	#print len(data.ravel()[:, np.newaxis]) # 16384
	#print proj
	#print len(proj)
	#print len(proj[0])


	#print len(data) #data是128 * 128的True-False矩阵, 其实就是原始图片
	#print len(data[0])
	#print set(data.ravel())
	#print data[123]

	#print len(proj) #proj是2304 * 1的向量
	#print len(proj[0])

	#print proj.ravel()


	#plt.imshow(data, cmap = plt.cm.gray)
	#plt.show()


	#Reconstruction with L2 (Ridge) Penalization
	rgr_ridge = Ridge(alpha = 0.2)
	rgr_ridge.fit(proj_operator, proj.ravel()) #这里X_train是 一个稀疏矩阵，y_train是一个2304 * 1 的double向量
	rec_l2 = rgr_ridge.coef_.reshape(l, l)      #训练出来的结果是一个16384 * 1的向量 -- 系数？ 这里面的意义是什么？ 这些系数代表了一张图片？

	#Reconstruction with L1 (Lasso) Penalization
	#这里的aphla是根据交叉验证获得的 LassoCV

	rgr_lasso = Lasso(alpha = 0.001)
	rgr_lasso.fit(proj_operator, proj.ravel())
	rec_l1 = rgr_lasso.coef_.reshape(l, l)

	print set(rgr_lasso.coef_)
	#print len(rgr_lasso.coef_)  #rgr_lasso.coef_是一个16384 * 1的向量， 其实就是对应于原图的所有像素变成了一个向量


	#plt.imshow(rec_l1, cmap = plt.cm.gray)
	#plt.show()



	plt.figure(figsize = (8, 3.3))

	plt.subplot(131)
	plt.imshow(data, cmap=plt.cm.gray, interpolation = "nearest")
	plt.axis('off')
	plt.title('Original image')

	plt.subplot(132)
	plt.imshow(rec_l2, cmap = plt.cm.gray, interpolation = "nearest")
	plt.axis('off')
	plt.title('L2 penalization')

	plt.subplot(133)
	plt.imshow(rec_l1, cmap = plt.cm.gray, interpolation = "nearest")
	plt.axis('off')
	plt.title('L1 penalization')

	plt.subplots_adjust(hspace = 0.01, wspace = 0.01, top = 1, bottom = 0, left = 0, right = 1)

	plt.show()



def spectralClusteringTest01():
	import numpy as np
	import matplotlib.pyplot as plt

	from sklearn.feature_extraction import image
	from sklearn.cluster import spectral_clustering

	l = 100
	x,y = np.indices((l, l)) #x,y 都是二维矩阵， 表示了某点的x 和 y的坐标


	center1 = (28, 24)
	center2 = (40, 50)
	center3 = (67, 58)
	center4 = (24, 70)

	radius1, radius2, radius3, radius4 = 16, 14, 15, 14

	circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
	circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
	circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
	circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2


	img = circle1 + circle2 + circle3 + circle4
	mask = img.astype(bool)
	img = img.astype(float)

	img += 1 + 0.2 * np.random.randn(*img.shape)

	#Convert the image into a graph with the value of the gradient on the edges

	#img就是一个100 * 100的图片
	#mask是一个bool型的100 * 100模板
	#graph是一个稀疏矩阵 -- 不过为什么是2678 * 2678 ?
	#估计这一步里面计算了梯度
	graph = image.img_to_graph(img, mask = mask)

	print graph.shape
	graph.data = np.exp(-graph.data / graph.data.std())

	#这里还是指定了聚类的中心数目
	#这里是只对mask内的点进行聚类
	labels = spectral_clustering(graph, n_clusters = 4, eigen_solver = "arpack")


	print labels

	label_im = -np.ones(mask.shape)
	label_im[mask] = labels

	plt.matshow(img)
	plt.matshow(label_im)

	plt.show()




def imageDenoisingTest01():
	from time import time
	import matplotlib.pyplot as plt
	import numpy as np

	from scipy.misc import lena

	from sklearn.decomposition import MiniBatchDictionaryLearning
	from sklearn.feature_extraction.image import extract_patches_2d
	from sklearn.feature_extraction.image import reconstruct_from_patches_2d

	#Load image and extract patches
	lena = lena() / 256.0




	lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
	lena /= 4.0

	height, width = lena.shape

	#Distort the right half of the image
	print "distorting image"

	distorted = lena.copy()
	distorted[:, height//2:] += 0.075 * np.random.randn(width, height // 2)

	#plt.imshow(distorted[:, :height//2], cmap = plt.cm.gray, interpolation = "nearest")
	#plt.show()

	print "Extacting reference patches"
	#这里是从distorted的左半边抽取patches
	t0 = time()
	patch_size = (7, 7)
	data = extract_patches_2d(distorted[:, :height//2], patch_size)

	#data是 30500 * 7 * 7 维矩阵
	#print data
	#print len(data)
	#print len(data[0][0])

	#plt.imshow(data[0], cmap = plt.cm.gray, interpolation = "nearest")
	#plt.show()

	#print distorted[:, height//2:].shape #一半是256 * 128




	#下面是把patch转换为一维向量, 然后再归一化
	data = data.reshape(data.shape[0], -1)
	data -= np.mean(data, axis = 0)
	data /= np.std(data, axis = 0)

	print 'done in ' + str(time() - t0)


	# Learn the dictionary from reference patches
	print "Learning the dictionary"
	t0 = time()
	#这一步是开始对patches进行学习
	#new 一个model
	dico = MiniBatchDictionaryLearning(n_components = 100, alpha = 1, n_iter = 5000)

	print data.shape  #data是30500 * 49维矩阵
	V = dico.fit(data).components_

	print V.shape #V是100 * 49维矩阵
	dt = time() - t0

	print "done in %.2fs." % dt

	plt.figure(figsize = (4.2, 4))
	for i, comp in enumerate(V[:100]):
		plt.subplot(10, 10, i + 1)
		plt.imshow(comp.reshape(patch_size), cmap = plt.cm.gray_r, interpolation = "nearest")
		plt.xticks(())
		plt.yticks(())

	plt.suptitle("Dictionary learned from lena patches\n" + "Train time %.1fs on %d patches" % (dt, len(data)), fontsize = 16)

	plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

	def show_with_diff(image, reference, title):
		plt.figure(figsize = (5, 3.3))
		plt.subplot(1, 2, 1)
		plt.title('Image')
		plt.imshow(image, vmin = 0, vmax = 1, cmap = plt.cm.gray, interpolation = "nearest")

		plt.xticks(())
		plt.yticks(())
		plt.subplot(1,2,2)

		difference = image - reference

		plt.title("difference (norm: %.2f)" % np.sqrt(np.sum(difference ** 2)))

		plt.imshow(difference, vmin = -0.5, vmax = 0.5, cmap = plt.cm.PuOr, interpolation = "nearest")
		plt.xticks(())
		plt.yticks(())
		plt.suptitle(title, size = 16)

		plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.02)


	show_with_diff(distorted, lena, "Distorted Image")




	#plt.show()

	#Extract noisy patches and reconstruct them using the dictionary
	#从右半边抽取patches
	print('Extracting noisy pathces...')
	t0 = time()
	data = extract_patches_2d(distorted[:, height//2:], patch_size)
	data = data.reshape(data.shape[0], -1)
	intercept = np.mean(data, axis = 0)
	data -= intercept

	print "done in %.2fs. " % (time() - t0)

	transform_algorithms = [('Orthogonal Matching Pursuit\n1 atom', 'omp',
							{'transform_n_nonzero_coefs': 1}),
							('Orthogonal Matching Pursuit\n2 atoms', 'omp',
							{'transform_n_nonzero_coefs': 2}),
							('Least-angle regression\n5 atoms', 'lars',
							{'transform_n_nonzero_coefs': 5}),
							('Thresholding\n alpha = 0.1', 'threshold',
							{'transform_alpha': 0.1})]

	reconstructions = {}
	for title, transform_algorithm, kwargs in transform_algorithms:
		print title + "..."
		reconstructions[title] = lena.copy()
		t0 = time()
		dico.set_params(transform_algorithm = transform_algorithm, **kwargs)
		code = dico.transform(data) #利用之前训练的模型来获得代表系数 -- code
		patches = np.dot(code, V)

		if transform_algorithm == "threshold":
			patches -= patches.min()
			patches /= patches.max()

		patches += intercept
		patches = patches.reshape(len(data), *patch_size)

		if transform_algorithm == "threshold":
			patches -= patches.min()
			patches /= patches.max()

		reconstructions[title][:, height // 2:] = reconstruct_from_patches_2d(patches, (width, height // 2))
		dt = time() - t0
		print "done in %.2fs." % dt
		show_with_diff(reconstructions[title], lena, title + '(time: %.1fs)' % dt)

	plt.show()




def SwissRollTest01():
	import matplotlib.pyplot as plt

	from mpl_toolkits.mplot3d import Axes3D

	# Locally Linear Embedding of the swiss roll
	from sklearn import manifold, datasets
	#这里是生成数据集 X是1500 * 3 的矩阵(表示location)，color是1500 * 1的矩阵(表示label--颜色)
	X, color = datasets.samples_generator.make_swiss_roll(n_samples = 1500)

	#print X[1,:]
	#print color[1]


	X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2)
	#X_r是1500 * 2的矩阵，err是一个实数。
	print X_r.shape
	#print err

	fig = plt.figure()

	try:
		ax = fig.add_subplot(211, projection='3d')
		ax.scatter(X[:,0], X[:,1], X[:,2], c=color, cmap=plt.cm.Spectral)
	except:
		ax = fig.add_subplot(211)
		ax.scatter(X[:,0],X[:,2],c=color, cmap=plt.cm.Spectral)
		print ""
	ax.set_title("Original data")
	ax = fig.add_subplot(212)

	#这里绘制的是projected data
	ax.scatter(X_r[:,0], X_r[:,1], c=color, cmap=plt.cm.Spectral)

	plt.axis('tight')
	plt.xticks([]), plt.yticks([])
	plt.title('Projected Data')
	plt.show()



def plot_lle_digis():

	from time import time
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import offsetbox
	from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection)

	digits = datasets.load_digits(n_class = 6)  #这里load的只有数字0-5
	X = digits.data
	y = digits.target

	n_samples, n_features = X.shape #(1083 * 64)
	n_neighbors = 30

	#print X.shape
	#print set(y)


	def plot_embedding(X, title=None):
		#这个函数输入的是X坐标(二维)和title
		x_min, x_max = np.min(X, 0), np.max(X,0)
		X = (X - x_min) / (x_max - x_min)

		plt.figure()
		ax = plt.subplot(111)
		for i in range(X.shape[0]):
			plt.text(X[i,0], X[i,1], str(digits.target[i]), color = plt.cm.Set1(y[i] / 10.0), fontdict={'weight':'bold', 'size':9})

		if hasattr(offsetbox, "AnnotationBbox"):
			shown_images = np.array([[1.0, 1.0]])
			for i in range(digits.data.shape[0]):
				dist = np.sum((X[i] - shown_images) ** 2, 1)
				if np.min(dist) < 4e-3:
					continue
				shown_images = np.r_[shown_images, [X[i]]]
				imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i])

				ax.add_artist(imagebox)

		plt.xticks([]), plt.yticks([])

		if title is not None:
			plt.title(title)


	#Plot images of the digits

	n_img_per_row = 20
	img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row)) #这里是由N张小图像生成一张大图像
	for i in range(n_img_per_row):
		ix = 10 * i + 1
		for j in range(n_img_per_row):
			iy = 10 * j + 1
			img[ix: ix + 8, iy : iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

	plt.imshow(img, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	plt.title('A selection from the 64-dimensional digits dataset')


	#----------------------------------------------------------------------
	#Random 2D prjection using a random unitary matrix
	print "Computing random projection"
	rp = random_projection.SparseRandomProjection(n_components=2, random_state = 42)
	X_projected = rp.fit_transform(X)  #X_projected 是 1083 * 2的矩阵
	#print X_projected.shape

	plot_embedding(X_projected, "Random Projection of the digits")

	#----------------------------------------------------------------------
	#Projection on to the first principle components
	print "Computing PCA projection"
	t0 = time()
	#这里用的是decomposition.TruncatedSVD这个model来进行pca -- 学一下SVD
	#X_pca也是1083 *2的矩阵
	X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
	plot_embedding(X_pca, "Principle Components projection of the digits (time %.2fs)" % (time() - t0))

	#print X_pca.shape


	#----------------------------------------------------------------------
	#projection on to the first 2 linear discriminant components
	print "Computing LDA projections"
	X2 = X.copy()
	X2.flat[::X.shape[1] + 1] += 0.01 #make X invertible -- 这个flat不知道是什么意思...
	t0 = time()

	#X2 与X的shape相同
	X_lda = lda.LDA(n_components=2).fit_transform(X2, y)
	plot_embedding(X_lda, "Linear Discriminant projection of the digits (time %.2fs)" % (time() - t0))

	#print X2.shape
	#print X_lda.shape



	#----------------------------------------------------------------------
	#Isomap projection of the digits dataset
	print "Computing Isomap embedding"

	t0 = time()
	X_iso = manifold.Isomap(n_neighbors, n_components = 2).fit_transform(X)
	print("Done.")
	plot_embedding(X_iso, "Iso projection of the digits (time %.2fs)" % (time() - t0))


	#----------------------------------------------------------------------
	#Locally linear embedding of the digits dataset
	print "Computing LLE embedding"
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method = "standard")
	t0 = time()
	X_lle = clf.fit_transform(X)
	print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
	plot_embedding(X_lle, "LLE standard (time %.2fs)" % (time() - t0))

	#----------------------------------------------------------------------
	#HLLE embedding of the digits dataset
	print "Computing Hessian LLE embedding"
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method = "hessian")
	t0 = time()
	X_hlle = clf.fit_transform(X)
	print "Done. Reconstruction error: %g" % clf.reconstruction_error_
	plot_embedding(X_hlle, "Hessian LLE (time %.2fs)" % (time() - t0))


	#----------------------------------------------------------------------
	#LTSA embedding of the digits dataset
	print "Computing LTSA embedding"
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method = "ltsa")
	t0 = time()
	X_ltsa = clf.fit_transform(X)
	print "Done. Reconstruction error: %g" % clf.reconstruction_error_
	plot_embedding(X_ltsa, "LTSA (time %.2fs)" % (time() - t0))


	#----------------------------------------------------------------------
	#MDS embedding of the digits dataset

	print "Computing MDS embedding"
	clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
	t0 = time()

	X_mds = clf.fit_transform(X)
	print "Done. Stress: %f" % clf.stress_
	plot_embedding(X_mds, "MDS embedding (time %.2fs)" % (time() - t0))

	#----------------------------------------------------------------------
	#Random Trees embedding of the digits datasets
	#这里是先进行RandomTreesEmbedding再进行pca
	print "Computing Totally random trees embedding"
	hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state = 0, max_depth=5)
	t0 = time()

	X_transformed = hasher.fit_transform(X)
	pca = decomposition.TruncatedSVD(n_components=2)
	X_reduced = pca.fit_transform(X_transformed)

	plot_embedding(X_reduced, "Random Trees embedding (time %.2fs)" % (time() - t0))


	#----------------------------------------------------------------------
	#Spectral embedding of the digits dataset
	print "Computing Spectral embedding"
	embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")

	t0 = time()
	X_se = embedder.fit_transform(X)

	plot_embedding(X_se, "Spectral embedding (time %.2fs)" % (time() - t0))


	#----------------------------------------------------------------------

	#t-SNE embedding of the digits dataset
	print "Computing t-SNE embedding"
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	t0 = time()
	X_tsne = tsne.fit_transform(X)

	plot_embedding(X_tsne, "TSNE embedding (time %.2fs)" % (time() - t0))

	plt.show()




def labelPropagationTest01():
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.semi_supervised import label_propagation
	from sklearn.datasets import make_circles

	#Generate Data
	n_samples = 200
	X,  y = make_circles(n_samples = n_samples, shuffle = False)
	outer, inner = 0, 1
	labels = -np.ones(n_samples)

	#这里是设置label,相当于是给了一个初始值， 其他的label都不知道
	labels[0] = outer
	labels[-1] = inner

	#print X.shape #X是200 * 2的矩阵
	#print labels #label头尾分别是0和1， 其他全部是-1
	#print y #前100个是0， 后100个是1

	#Learn with LabelSpreading
	#这里输入的是X和labels -- 没有输入y
	#但是这里的labels是要预测的量
	label_spread = label_propagation.LabelSpreading(kernel = "knn", alpha = 1.0)
	label_spread.fit(X, labels)

	#Plot output labels
	output_labels = label_spread.transduction_
	plt.figure(figsize=(8.5, 4))
	plt.subplot(1,2,1)
	plot_outer_labeled, = plt.plot(X[labels == outer, 0], X[labels == outer, 1], 'rs')
	plot_unlabeled, = plt.plot(X[labels == -1, 0], X[labels == -1, 1], 'g.')
	plot_inner_labeled, = plt.plot(X[labels == inner, 0], X[labels == inner, 1], 'bs')

	plt.legend((plot_outer_labeled, plot_inner_labeled, plot_unlabeled), ("Outer Labeled", "Inner Labeled", "Unlabeled"), "upper left", numpoints = 1, shadow = False)
	plt.title("Raw data (2 classes = red and blue)")


	#要学习这里的画图技巧
	plt.subplot(1,2,2)
	output_label_array = np.asarray(output_labels)
	outer_numbers = np.where(output_label_array == outer)[0]
	inner_numbers = np.where(output_label_array == inner)[0]
	plot_outer, = plt.plot(X[outer_numbers, 0], X[outer_numbers, 1], 'rs')
	plot_inner, = plt.plot(X[inner_numbers, 0], X[inner_numbers, 1], 'bs')

	plt.legend((plot_outer, plot_inner), ('Outer Learned', 'Inner Learned'), 'upper left', numpoints = 1, shadow = False)

	plt.title("Labels Learned with Label Spreading (KNN)")
	plt.subplots_adjust(left = 0.07, bottom = 0.07, right = 0.93, top = 0.92)

	plt.show()


def plot_stock_market():
	import datetime
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib import finance
	from matplotlib.collections import LineCollection
	
	from sklearn import cluster, covariance, manifold
	
	d1 = datetime.datetime(2003, 1, 1)
	d2 = datetime.datetime(2008, 1, 1)
	
	symbol_dict = {
		'TOT': 'Total',
		'XOM': 'Exxon',
		'CVX': 'Chevron',
		'COP': 'ConocoPhillips',
		'VLO': 'Valero Energy',
		'MSFT': 'Microsoft',
		'IBM': 'IBM',
		'TWX': 'Time Warner',
		'CMCSA': 'Comcast',
		'CVC': 'Cablevision',
		'YHOO': 'Yahoo',
		'DELL': 'Dell',
		'HPQ': 'HP',
		'AMZN': 'Amazon',
		'TM': 'Toyota',
		'CAJ': 'Canon',
		'MTU': 'Mitsubishi',
		'SNE': 'Sony',
		'F': 'Ford',
		'HMC': 'Honda',
		'NAV': 'Navistar',
		'NOC': 'Northrop Grumman',
		'BA': 'Boeing',
		'KO': 'Coca Cola',
		'MMM': '3M',
		'MCD': 'Mc Donalds',
		'PEP': 'Pepsi',
		'MDLZ': 'Kraft Foods',
		'K': 'Kellogg',
		'UN': 'Unilever',
		'MAR': 'Marriott',
		'PG': 'Procter Gamble',
		'CL': 'Colgate-Palmolive',
		'GE': 'General Electrics',
		'WFC': 'Wells Fargo',
		'JPM': 'JPMorgan Chase',
		'AIG': 'AIG',
		'AXP': 'American express',
		'BAC': 'Bank of America',
		'GS': 'Goldman Sachs',
		'AAPL': 'Apple',
		'SAP': 'SAP',
		'CSCO': 'Cisco',
		'TXN': 'Texas instruments',
		'XRX': 'Xerox',
		'LMT': 'Lookheed Martin',
		'WMT': 'Wal-Mart',
		'WAG': 'Walgreen',
		'HD': 'Home Depot',
		'GSK': 'GlaxoSmithKline',
		'PFE': 'Pfizer',
		'SNY': 'Sanofi-Aventis',
		'NVS': 'Novartis',
		'KMB': 'Kimberly-Clark',
		'R': 'Ryder',
		'GD': 'General Dynamics',
		'RTN': 'Raytheon',
		'CVS': 'CVS',
		'CAT': 'Caterpillar',
		'DD': 'DuPont de Nemours'}
	
	symbols, names = np.array(list(symbol_dict.items())).T
	#quotes = [finance.quotes_historical_yahoo(symbol, d1, d2, asobject = True) for symbol in symbols]
	
	#print quotes
	print "This function is not finished!"
	
	
	
def random_forest_embedding():
	import numpy as np
	import matplotlib.pyplot as plt
	
	from sklearn.datasets import make_circles
	from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
	from sklearn.decomposition import TruncatedSVD
	from sklearn.naive_bayes import BernoulliNB
	
	#建立数据集
	X, y = make_circles(factor = 0.5, random_state = 0, noise = 0.05)
	
	#print y
	#print X.shape #X 是100 * 2, y是100 * 1 (0,1数组)
	
	
	#Transform data
	hasher = RandomTreesEmbedding(n_estimators = 10, random_state = 0, max_depth = 3) #设置参数，生成model
	X_transformed = hasher.fit_transform(X)
	
	#print X_transformed[99]
	#print X_transformed.shape #100 * 74 ? 可能是如下原因 -- 为什么利用高维稀疏表示之后可以有助于分类？
	#RandomTreesEmbedding provides a way to map data to a very high-dimensional, 
	#sparse representation, which might be beneficial for classification. 
	
	pca = TruncatedSVD(n_components = 2)
	X_reduced = pca.fit_transform(X_transformed)
	
	#print X_reduced #这里是X_reduced 是 100 * 2

	#Learn a Naive bayes classifier on the transformed data
	nb = BernoulliNB()
	nb.fit(X_transformed, y) #利用高维稀疏矩阵和y进行训练
	
	#Learn a ExtraTreesClassifier for comparison
	trees = ExtraTreesClassifier(max_depth = 3, n_estimators = 10, random_state = 0)
	trees.fit(X, y) #这里是利用原始的2维X和y进行训练
	
	#scatter plot of original and reduced data
	fig = plt.figure(figsize = (9, 8))
	ax = plt.subplot(221)
	ax.scatter(X[:, 0], X[:, 1], c = y, s = 50) #X[:, 0]是X坐标 X[:, 1]是Y坐标， y是label
	ax.set_title("Original Data(2d)")
	ax.set_xticks(())
	ax.set_yticks(())
	
	ax = plt.subplot(222)
	#注意虽然X在转化之后了，但是对应的label没有变，所以可以根据label来分析transfrom的效果
	ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c = y, s = 50) 
	ax.set_title("pca reduction (2d) of transformed data (%dd)" % X_transformed.shape[1]) 
	ax.set_xticks(())
	ax.set_yticks(())
	
	
	
	#Plot the decision in original space
	h = 0.01
	x_min, x_max = X[:, 0].min() - 0.5, X[:,0].max() + 0.5
	y_min, y_max = X[:, 1].min() - 0.5, X[:,1].max() + 0.5
	
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	
	#transform grid using RandomTreesEmbedding
	#利用nb来做predict
	transformed_grid = hasher.transform(np.c_[xx.ravel(), yy.ravel()])
	y_grid_pred = nb.predict_proba(transformed_grid)[:, 1]
	
	
	ax = plt.subplot(223)
	ax.set_title("Naive Bayes on Transformed data")
	ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
	ax.scatter(X[:, 0], X[:, 1], c = y, s = 50) #X[:, 0]是X坐标 X[:, 1]是Y坐标， y是label
	
	ax.set_ylim(-1.4, 1.4)
	ax.set_xlim(-1.4, 1.4)
	ax.set_xticks(())
	ax.set_yticks(())
	
	
	#transform grid using ExtraTreesClassifier
	#利用trees做predict
	y_grid_pred = trees.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
	
	ax = plt.subplot(224)
	ax.set_title("ExtraTrees predictions")
	ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
	ax.scatter(X[:, 0], X[:, 1], c = y, s = 50) #X[:, 0]是X坐标 X[:, 1]是Y坐标， y是label
	
	ax.set_ylim(-1.4, 1.4)
	ax.set_xlim(-1.4, 1.4)
	ax.set_xticks(())
	ax.set_yticks(())

	plt.tight_layout()
	plt.show()

















