import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def LoadData():
	print("Loading data ... ")

	data = np.loadtxt("data.csv", delimiter='\t', skiprows=1)
	data = data[:,1:]
	return data

def BuildXYPair(inputData):
	x = inputData[:, :-1]
	y = inputData[:, -1]
	return (x,y)

def TrainData(inputData):
	# put all parameters here
	frac_split = 0.7

	# shuffle data
	np.random.shuffle(inputData)

	# split data
	split_index = int(inputData.shape[0]*frac_split)
	inputData_train = inputData[0:split_index, :]
	inputData_test = inputData[split_index:, :]

	# convert to X,Y pair
	X_train, Y_train = BuildXYPair(inputData_train)
	X_test, Y_test = BuildXYPair(inputData_test)

	# build classifier
	bdt = AdaBoostClassifier(
		                      DecisionTreeClassifier(max_depth=3),
		                      n_estimators=800,
		                      learning_rate=1.,
		                    )

	# training
	print("Training BDT ... ")
	bdt.fit(X_train, Y_train)

	return (bdt, inputData_test)

def TestModel(model, TestData):
	X_test, Y_test = BuildXYPair(TestData)

	# test
	# x = X_test[30:31,:]
	# y = Y_test[30:31]

	# print(x)
	# print(y)
	# print(model.predict(x))
	# print(model.predict_log_proba(x))
	# print(model.predict_proba(x))

	print("--------------------")

	print(model.score(X_test, Y_test))

def test():
	data = LoadData()
	bdt, TestData = TrainData(data)
	TestModel(bdt, TestData)


