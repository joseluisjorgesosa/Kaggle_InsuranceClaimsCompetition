import simpleNN as snn
import random_parameters as rp
import sys
import numpy as np
import xgboost as xgb
import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import linear_model

#Function: load_data
#
#Description: loads trianing data into global vars
#
#
def load_data(filename):
	return  snn.get_training_data(filename)

# Function: get_neural_net_paramters
#
# Description: Gets the parameters for nueral networks from csv file
#

def get_neural_networks(network_paramter_file):
	population = []

 	network_file = open(network_paramter_file,'r')


	#Initialize a population of neural nets
	for line in network_file:
		opts = rp.get_default_opts()

		parameters = line.split(",")

 		opts['step_size'] = float(parameters[0])
 		opts['momentum']   = float(parameters[1])
 		opts['num_epochs'] = int(parameters[2])
		opts['batch_size'] = int(parameters[3])
 		opts['hidden_units_1'] = int(parameters[4])
		opts['hidden_units_2'] = int(parameters[5])
		opts['dropout_rate_1'] = float(parameters[6])
		opts['dropout_rate_2'] = float(parameters[7])
		opts['activation_func_1'] = parameters[8]
		opts['activation_func_2'] = parameters[9].rstrip('\n')


		population.append(opts)

	return population

#Function: train the nueral network
#
#Description: Get a network and input var variable for each set of parameters
#
def train_neural_networks(X_train_groups, y_train, X_val_groups, y_val, df):
	networks = []
	i=0
	for i in range(len(X_train_groups)):
		X_train = X_train_groups[i]
		X_val = X_val_groups[i]
		network,input_var = snn.get_neural_net_default(X_train, y_train, X_val, y_val, df)

		pair = []
		pair.insert(0,network)
		pair.insert(1,input_var)

		networks.insert(i,pair)

	return networks

#Function: make predictions
#
#Description: Make predictions for each neural network and append them as a column to a matrix
#
#
def make_predictions(networks,X_test_groups):

	new_training_data = np.zeros((len(X_test_groups[0]),len(networks)))

	print("Making predictions...")
	#Get predictions from neural networks
	feat = 0;

	for i in range(len(X_test_groups)):
		pair = networks[i]

		network = pair[0]
		input_var = pair[1]

		print(input_var)
		print(X_test_groups[i].shape)
		predictions = snn.get_predictions(network,input_var, X_test_groups[i])

		new_training_data[0:,feat] = predictions[0:,1]

		feat += 1

	return new_training_data

#Function: make_data_groups
#
#Description: Split the data up into groups of k features
#
def make_feature_groups(k,X_train,X_val,X_test):

	numFeatures = len(X_train[0,0,0:])
	groupIndices = []

	i = 0
	while(min(i+k,numFeatures) - i > 0):
		start_stop = []
		start_stop.insert(0,i)
		start_stop.insert(1,min(i+k,numFeatures))

		groupIndices.append(start_stop)
		i = i + k

	X_train_groups = []
	X_val_groups = []
	X_test_groups = []

	for start_stop in groupIndices:
		print(start_stop)
		start = start_stop[0]
		stop = start_stop[1]

		Xtrn = np.zeros((len(X_train),1,stop-start))
		Xvl	= np.zeros((len(X_val),1,stop-start))
		Xtst  = np.zeros((len(X_test),1,stop-start))

		Xtrn[0:,0,0:] = X_train[0:,0,start:stop]
		Xvl[0:,0,0:] = X_val[0:,0,start:stop]
		Xtst[0:,0,0:] = X_test[0:,0,start:stop]

		X_train_groups.append(Xtrn)
		X_val_groups.append(Xvl)
		X_test_groups.append(Xtst)


	return X_train_groups, X_val_groups, X_test_groups	





#Function: write_preds_to_file
#
#Description:  Writes preidcitons to a file
#
def write_preds_to_file(test_submission_file, preds):

	test_submission = open(test_submission_file,'w')
	test_submission.write('ID,PredictedProb\n')

	for i in range(len(preds)):
		test_submission.write(str(ids_test[i]) + "," + str(preds[i])+"\n")

	test_submission.close()

def get_predictions(trainFile,testFile):
	print("Loading training data...")
	X_train, y_train, X_val, y_val, df = load_data(trainFile)

	print("Loading testing data...")
	X_test,ids_test = snn.get_test_data(testFile)

	k= 2
	X_train_groups, X_val_groups, X_test_groups = make_feature_groups(k,X_train,X_val,X_test)


	print("Building networks...")
	networks = train_neural_networks(X_train_groups, y_train, X_val_groups, y_val, df)

	#Make predictions
	print("Making predictions...")
	train_predictions = make_predictions(networks,X_train_groups)
	val_predictions = make_predictions(networks,X_val_groups)
	test_predictions = make_predictions(networks,X_test_groups)

	#Put all numpy arrays into data grames
	train_predictions = pd.DataFrame(train_predictions)
	val_predictions  = pd.DataFrame(val_predictions)
	test_predictions = pd.DataFrame(test_predictions)
	y_train = pd.DataFrame(y_train)
	y_val = pd.DataFrame(y_val)

	print("Training XGBoost...")
	#Do XGboosting on predictions
	train_DMatrix = xgb.DMatrix(train_predictions, y_train, missing = -999)
	test_DMatrix = xgb.DMatrix(val_predictions, y_val, missing = -999) #for validation/early stopping


	predict_DMatrix = xgb.DMatrix(test_predictions, missing = -999)


	#specify parameters
	param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss'}

	#specify validations set to watch performance
	watchlist  = [(train_DMatrix,'train'),(test_DMatrix,'eval')]
	num_round = 5000
	bst = xgb.train(param, train_DMatrix, num_round, watchlist, early_stopping_rounds=15)


	print("Making XGBoost predictions...")
	#prediction
	train_preds = bst.predict(train_DMatrix)
	val_preds = bst.predict(test_DMatrix)
	test_preds = bst.predict(predict_DMatrix)

	return train_preds, val_preds, test_preds


if __name__ == "__main__":

	#Load training data
	print("Loading training data...")
	X_train, y_train, X_val, y_val, df = load_data(sys.argv[1])


	#Load testing data
	print("Loading testing data...")
	X_test,ids_test = snn.get_test_data(sys.argv[2])

	#Split data into groups of k features
	k= 2
	X_train_groups, X_val_groups, X_test_groups = make_feature_groups(k,X_train,X_val,X_test)

	#Build networks
	print("Building networks...")
	networks = train_neural_networks(X_train_groups, y_train, X_val_groups, y_val, df)
		
	#Make predictions
	print("Making predictions...")
	train_predictions = make_predictions(networks,X_train_groups)
	val_predictions = make_predictions(networks,X_val_groups)
	test_predictions = make_predictions(networks,X_test_groups)


	#Put all numpy arrays into data grames
	train_predictions = pd.DataFrame(train_predictions)
	val_predictions  = pd.DataFrame(val_predictions)
	test_predictions = pd.DataFrame(test_predictions)
	y_train = pd.DataFrame(y_train)
	y_val = pd.DataFrame(y_val)

	print("Training XGBoost...")
	#Do XGboosting on predictions
	train_DMatrix = xgb.DMatrix(train_predictions, y_train, missing = -999)
	test_DMatrix = xgb.DMatrix(val_predictions, y_val, missing = -999) #for validation/early stopping


	predict_DMatrix = xgb.DMatrix(test_predictions, missing = -999)


	#specify parameters
	param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss'}

	#specify validations set to watch performance
	watchlist  = [(train_DMatrix,'train'),(test_DMatrix,'eval')]
	num_round = 5000
	bst = xgb.train(param, train_DMatrix, num_round, watchlist, early_stopping_rounds=15)


	print("Making XGBoost predictions...")
	#prediction
	preds = bst.predict(predict_DMatrix)

	print("Writing final predictions to file")
	results_file = sys.argv[3]+ "_k_3" + "_Depth_6"+"_.csv"
	write_preds_to_file(results_file, preds)


