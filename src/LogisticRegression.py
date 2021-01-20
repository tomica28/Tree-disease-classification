"""
Implementation of Logistic Regression Algorithm in classifying diseased trees
"""

import math
import time
import numpy as np
import pandas as pd
from numpy import where
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
#
df_train = pd.read_csv("normalized_train_set.csv", header=0)
df_test = pd.read_csv("normalized_test_set.csv", header=0)

# print(df_train.head(10))
# print(df_train.shape)
# print(df_test.head(10))
# print(df_test.shape)

# split dataset inputs and output from [training.csv]
X_train = df_train.iloc[:,1:]
X_train = np.array(X_train)
Y_train = df_train.iloc[:,0]
Y_train = np.array(Y_train)

# # split dataset inputs and output from [testing.csv]
# X_train = df_test.iloc[:,1:]
# X_train = np.array(X_train)
# Y_train = df_test.iloc[:,0]
# Y_train = np.array(Y_train)
#
# # creating testing and training set
X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.33, shuffle=True)

# # split testing inputs and output from [testing.csv]
# X_test = df_test.iloc[:,1:]
# X_test = np.array(X_test)
# Y_test = df_test.iloc[:,0]
# Y_test = np.array(Y_test)

#train scikit learn model
init_alpha = 0.1
init_ITERA = 100
TRIES_ALPHA = 20
TRIES_ITER = 20
iterations = [None]*TRIES_ITER
alpha = [None]*TRIES_ALPHA
for i in range (TRIES_ITER):
    iterations[i] = init_ITERA
    init_ITERA += 50
for j in range(TRIES_ALPHA):
    alpha[j] = init_alpha
    init_alpha +=0.1

param_grid = dict(max_iter=iterations,C=alpha)
clf = LogisticRegression(penalty='l2');
random = RandomizedSearchCV(estimator=clf, param_distributions=param_grid)
start_time = time.time()
random_result = random.fit(X_train, Y_train)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
#print("Execution time: " + str((time.time() - start_time)) + ' ms')
# update parameters
my_max_iter = random_result.best_params_["max_iter"]
my_C= random_result.best_params_["C"]

## check the accuracy
clf2 = LogisticRegression(penalty='l2', C=my_C, max_iter=my_max_iter)
clf2.fit(X_train,Y_train)
print ('score Scikit learn tuning: ', random_result.score(X_test,Y_test))

####################################################################################################
# visualize train data, uncomment "show()" to run it
pos = where(Y_train == 1)
neg = where(Y_train == 0)
scatter(X_train[pos, 1], X_train[pos, 2], marker='1', c='r')
scatter(X_train[neg, 1], X_train[neg, 2], marker='o', c='b')
xlabel('Train set - Exam the 1st feature')
ylabel('Train set - Exam the 2nd feature')
legend(['Diseased trees', 'Normal trees'])
# show()
#
# visualize test data, uncomment "show()" to run it
pos = where(Y_test == 1)
neg = where(Y_test == 0)
scatter(X_test[pos, 1], X_test[pos, 2], marker='1', c='r')
scatter(X_test[neg, 1], X_test[neg, 2], marker='o', c='b')
xlabel('Test set - Exam the 1st feature')
ylabel('Test set - Exam the 2nd feature')
legend(['Diseased trees', 'Normal trees'])
# show()
#
# visualize train data, uncomment "show()" to run it
pos = where(Y_train == 1)
neg = where(Y_train == 0)
scatter(X_train[pos, 1], X_train[pos, 4], marker='1', c='r')
scatter(X_train[neg, 1], X_train[neg, 4], marker='o', c='b')
xlabel('Train set - Exam the 1st feature')
ylabel('Train set - Exam the 4th feature')
legend(['Diseased trees', 'Normal trees'])
# show()
#
# visualize test data, uncomment "show()" to run it
pos = where(Y_test == 1)
neg = where(Y_test == 0)
scatter(X_test[pos, 1], X_test[pos, 4], marker='1', c='r')
scatter(X_test[neg, 1], X_test[neg, 4], marker='o', c='b')
xlabel('Test set - Exam the 1st feature')
ylabel('Test set - Exam the 4th feature')
legend(['Diseased trees', 'Normal trees'])
# show()
#
# visualize train data, uncomment "show()" to run it
pos = where(Y_train == 1)
neg = where(Y_train == 0)
scatter(X_train[pos, 2], X_train[pos, 4], marker='1', c='r')
scatter(X_train[neg, 2], X_train[neg, 4], marker='o', c='b')
xlabel('Train set - Exam the 2nd feature')
ylabel('Train set - Exam the 4th feature')
legend(['Diseased trees', 'Normal trees'])
# show()
#
# visualize test data, uncomment "show()" to run it
pos = where(Y_test == 1)
neg = where(Y_test == 0)
scatter(X_test[pos, 2], X_test[pos, 4], marker='1', c='r')
scatter(X_test[neg, 2], X_test[neg, 4], marker='o', c='b')
xlabel('Test set - Exam the 2nd feature')
ylabel('Test set - Exam the 4th feature')
legend(['Diseased trees', 'Normal trees'])
# show()

def Sigmoid(z):
	G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return G_of_Z

def Hypothesis(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)

def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		hi = Hypothesis(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	print ('cost is ', J )
	return J

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta

def Logistic_Regression(X,Y,alpha,theta,num_iters):
	m = len(Y)
	for x in range(num_iters):
		new_theta = Gradient_Descent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			Cost_Function(X,Y,theta,m)
			print ('theta ', theta)
			print ('cost is ', Cost_Function(X,Y,theta,m))
	Testing(theta)

def Testing(theta):
    score = 0
    length = len(X_test)
    for i in range(length):
        prediction = round(Hypothesis(X_test[i],theta))
        answer = Y_test[i]
        if prediction == answer:
            score += 1
    my_score = float(score) / float(length)

    print ('Accurancy= ', my_score)


#
# get dimension of features from dataset
dim = X_train.shape[1]
initial_theta = [None]*dim
for i in range(dim):
	initial_theta[i] = 0.0

alpha = 0.1
iterations = 50
# Logistic_Regression(X_train,Y_train,alpha,initial_theta,iterations)

