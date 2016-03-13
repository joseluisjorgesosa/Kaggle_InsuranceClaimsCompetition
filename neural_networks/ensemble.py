import simpleNN as snn
import random_parameters as rp
import sys
import numpy as np
import xgboost as xgb

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
def train_neural_networks(population,X_train, y_train, X_val, y_val, df):
	
	networks = []

	i=0
	for opts in population:
		network,input_var = snn.get_neural_net(opts, X_train, y_train, X_val, y_val, df)

		pair = []
		pair.insert(0,network)
		pair.insert(1,input_var)
		networks.append(pair)

	return networks

#Function: make predictions
#
#Description: Make predictions for each neural network and append them as a column to a matrix
#
#
def make_predictions(networks, X):
	
	new_training_data = np.zeros((len(X),len(networks)))

	feat = 0;
	for pair in networks:
		network = pair[0]
		input_var = pair[1]
		predictions = snn.get_predictions(network,input_var, X)

		new_training_data[0:,feat] = predictions[0:,1]

		feat += 1

	return new_training_data

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


if __name__ == "__main__":

	#Get parameters for the neural networks
	print("Getting neural network parameters...")
	parameters = get_neural_networks(sys.argv[1])

	#Load training data
	print("Loading training data...")
	X_train, y_train, X_val, y_val, df = load_data(sys.argv[2])

	#Load testing data
	print("Loading testing data...")
	X_test,ids_test = snn.get_test_data(sys.argv[3])

	#Build networks
	print("Building networks...")
	networks = train_neural_networks(parameters,X_train, y_train, X_val, y_val, df)

	#Make predictions
	print("Making predictions...")
	train_predictions = make_predictions(networks,X_train)
	val_predictions = make_predictions(networks,X_val)
	test_predictions = make_predictions(networks,X_test)

	print("Training XGBoost...")
	#Do XGboosting on predictions
	train_DMatrix = xgb.DMatrix(train_predictions, y_train,missing=NAN)
	test_DMatrix = xgb.DMatrix(val_predictions, y_val,missing=NAN) #for validation/early stopping


	predict_DMatrix = xgb.DMatrix(test_predictions,missing=NAN)

	#specify parameters
	param = {'max_depth':10, 'eta':0.1, 'silent':1, 'objective':'reg:linear' }

	#specify validations set to watch performance
	watchlist  = [(test_DMatrix,'eval'), (train_DMatrix,'train')]
	num_round = 70
	bst = xgb.train(param, train_DMatrix, num_round, watchlist)


	print("Making XGBoost predictions...")
	#prediction
	preds = bst.predict(predict_DMatrix)

	print("Writing final predictions to file")
	write_preds_to_file(sys.argv[4], preds)


