import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
import numpy as np


from sklearn import svm
from sklearn import linear_model

 #Function: load_data
def load_data(data_file):
	'''
	Load the training and testing data
	'''
	df = pd.read_csv(data_file)

	#separate id column
	id_col = df['ID']
	cols = df.columns.tolist()[1:]
	df = df[cols]

	# #replace missing values with -999
	df.fillna(-999, inplace=True)

	#encode categorical features as integers
	cat_columns = df.select_dtypes(['object']).columns
	for c in cat_columns:
		df[c] = df[c].astype('category')

	df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes) #this may not be strictly necessary, since we already convert to category type first

	
	return id_col, df

#Function: getTrainingData()
#
#Description: Loads the data and then puts it into numpy arrays
#
# Returns training and validation data to user
def get_training_data(training_file, validation_data=False): 
	if validation_data:
		ids_train, df = load_data(training_file)
		traindf, valdf = train_test_split(df, test_size = 0.2)

	else:

		ids_train, traindf = load_data(training_file)
		X_val, y_val  = None, None

	features = list(traindf.columns[1:])
	labels = traindf.columns[0]

	#Read in X training data
	X_train = np.zeros((len(traindf),len(features)))
	X_train[0:len(traindf),0:len(features)] = (np.asarray(traindf.iloc[0:len(traindf),1:])-np.mean(np.asarray(traindf.iloc[0:len(traindf),1:]),axis=0))/np.std(np.asarray(traindf.iloc[0:len(traindf),1:]),axis=0)
	X_train = np.array(X_train.astype('int64'))

	y_train = np.array(traindf[labels])
	y_train = y_train.astype('int64')
	
	if validation_data:
		#Read in X validation data
		X_val = np.zeros((len(valdf),len(features)))
		X_val[0:len(valdf),0:len(features)] = (np.asarray(valdf.iloc[0:len(valdf),1:])-np.mean(np.asarray(valdf.iloc[0:len(valdf),1:]),axis=0))/np.std(np.asarray(valdf.iloc[0:len(valdf),1:]),axis=0)
		X_val = np.array(X_val.astype('int64'))

		y_val = np.array(valdf[labels])
		y_val = y_val.astype('int64')

	if not validation_data:
		return X_train, y_train
	else: 
		return X_train, y_train, X_val, y_val

#Function: get_test_data
#
#Description: Loads the data and then puts it into numpy arrays
#
#Returns testing data to user
def get_test_data(test_data_filename):
	#Load testing data		 
	ids_test, testdf = load_data(test_data_filename)

	features = list(testdf.columns[0:])

	X_test = np.zeros((len(testdf),len(features)))
	X_test[0:len(testdf),0:len(features)] = (np.asarray(testdf.iloc[0:len(testdf),0:])-np.mean(np.asarray(testdf.iloc[0:len(testdf),0:]),axis=0))/np.std(np.asarray(testdf.iloc[0:len(testdf),0:]),axis=0)
	X_test = X_test.astype('int64')

	return X_test, ids_test





#Fucntion: write_results_file
#
#Description: Write the results from neural net predictions to a csv file
#
#
def write_results_file(results_filename, predictions, ids_test):
	results = open(results_filename,'w')

	results.write('ID,PredictedProb\n')

	for i in range(len(ids_test)):
		results.write(str(ids_test[i]) + "," + str(predictions[i][1]) + "\n")

	results.close()


if __name__ == "__main__":

	print("Loading training data...")
	X_train, y_train = get_training_data(sys.argv[1],False)


	print("Initializing SVM...")
	clf = linear_model.SGDClassifier(alpha=0.00001, average=True, class_weight=None, 
		epsilon=0.1,eta0=0.0, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
		 loss='log', n_iter=5, n_jobs=1, penalty='l1', power_t=0.5, random_state=None, 
		 shuffle=True, verbose=0, warm_start=False)

	print("Fitting SVM to training data...")
	clf.fit(X_train, y_train)


	print("Getting test data...")
	X_test, ids_test = get_test_data(sys.argv[2])

	print("Making predictions...")
	predictions = clf.predict_proba(X_test)

	print("Recording results...")
	write_results_file(sys.argv[3], predictions,ids_test)


