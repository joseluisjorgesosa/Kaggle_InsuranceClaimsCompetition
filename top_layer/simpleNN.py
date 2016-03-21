import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
import time 
import theano
import theano.tensor as T
import numpy as np
import lasagne
floatX = theano.config.floatX


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

	#replace missing values with -999
	df.fillna(0, inplace=True)

	#encode categorical features as integers
	cat_columns = df.select_dtypes(['object']).columns
	for c in cat_columns:
		df[c] = df[c].astype('category')

	df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes) #this may not be strictly necessary, since we already convert to category type first

	#Replace missing value with average value (by column)
	#col_labels = df.columns[0:]
	#for col_label in col_labels:
	#	df[col_label].fillna(df[col_label].mean(),inplace=True)
	

	return id_col, df

#Function: getTrainingData()
#
#Description: Loads the data and then puts it into numpy arrays
#
# Returns training and validation data to user
def get_training_data(training_file): 
	ids_train, df = load_data(training_file)

	traindf, valdf = train_test_split(df, test_size = .2, random_state = 1)

	features = list(traindf.columns[1:])
	labels = traindf.columns[0]

		
	#Read in X training data
	X_train = np.zeros((len(traindf),1,len(features)))
	X_train[0:len(traindf),0,0:len(features)] = (np.asarray(traindf.iloc[0:len(traindf),1:])-np.mean(np.asarray(traindf.iloc[0:len(traindf),1:]),axis=0))/np.std(np.asarray(traindf.iloc[0:len(traindf),1:]),axis=0)
	X_train = np.array(X_train.astype(floatX))

	y_train = np.array(traindf[labels])
	y_train = y_train.astype('int32')

	#Read in X validation data
	X_val = np.zeros((len(valdf),1,len(features)))
	X_val[0:len(valdf),0,0:len(features)] = (np.asarray(valdf.iloc[0:len(valdf),1:])-np.mean(np.asarray(valdf.iloc[0:len(valdf),1:]),axis=0))/np.std(np.asarray(valdf.iloc[0:len(valdf),1:]),axis=0)
	X_val = np.array(X_val.astype(floatX))

	y_val = np.array(valdf[labels])
	y_val = y_val.astype('int32')

	return X_train, y_train, X_val, y_val, df

#Function: get_test_data
#
#Description: Loads the data and then puts it into numpy arrays
#
#Returns testing data to user
def get_test_data(test_data_filename):
	#Load testing data		 
	ids_test, testdf = load_data(test_data_filename)

	features = list(testdf.columns[0:])

	X_test = np.zeros((len(testdf),1,len(features)))
	X_test[0:len(testdf),0,0:len(features)] = (np.asarray(testdf.iloc[0:len(testdf),0:])-np.mean(np.asarray(testdf.iloc[0:len(testdf),0:]),axis=0))/np.std(np.asarray(testdf.iloc[0:len(testdf),0:]),axis=0)
	X_test = X_test.astype(floatX)

	return X_test, ids_test


def build_mlp(numFeatures, opts):

	network = lasagne.layers.InputLayer(shape=(None,1,numFeatures),bias=lasagne.init.Constant(1))
	input_var = network.input_var

	network = lasagne.layers.BiasLayer(network, 
				b=lasagne.init.Constant(-1))


	network = lasagne.layers.DropoutLayer(network, p=opts['dropout_rate_1'], rescale=True)

	if(opts['activation_func_1'] == 'sigmoid'):
		activation_func_1 = lasagne.nonlinearities.sigmoid
	elif(opts['activation_func_1'] == 'tanh'):
		activation_func_1 =  lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=1)
	elif(opts['activation_func_1'] == 'linear'):
		activation_func_1 = lasagne.nonlinearities.linear
	elif(opts['activation_func_1'] == 'rectify'):
		activation_func_1 = lasagne.nonlinearities.rectify
	else:
			print(opts['activation_func_1'] + " not valid (1)")
			sys.exit(-1)

	network = lasagne.layers.DenseLayer(
			network, num_units=opts['hidden_units_1'],
			nonlinearity=activation_func_1,
			W=lasagne.init.GlorotNormal(gain=1))

	if(opts['two_hidden_layers'] == True):
		network = lasagne.layers.DropoutLayer(network, p=opts['dropout_rate_2'], rescale=True)

		if(opts['activation_func_2'] == 'sigmoid'):
			activation_func_2 = lasagne.nonlinearities.sigmoid
		elif(opts['activation_func_2'] == 'tanh'):
			activation_func_2 = lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=1)
		elif(opts['activation_func_2'] == 'linear'):
			activation_func_2 = lasagne.nonlinearities.linear
		elif(opts['activation_func_2'] == 'rectify'):
			activation_func_2 = lasagne.nonlinearities.rectify
		else:
			print(opts['activation_func_2'] + " not valid (2)")
			sys.exit(-1)

		network = lasagne.layers.DenseLayer(
					network, num_units=opts['hidden_units_2'],
					nonlinearity=activation_func_2,
					W=lasagne.init.GlorotNormal(gain=1))


	network = lasagne.layers.DenseLayer(
		network, num_units=2,
		nonlinearity=lasagne.nonlinearities.softmax)

	return input_var, network

# Function setup_network
# Description: Builds a neural net as specfied by input_var and opts. 
# Then sets up the prediciton, loss, update, training, and validation functions.
# 
# Returns network, training function, and validation function to user.
#
def setup_network(num_features, target_var, opts):

	input_var, network = build_mlp(num_features,opts)

	#create loss function 
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

	if 'alpha' in opts:
		loss = loss.mean() + opts['alpha']*lasagne.regularization.regularize_network_params( network, lasagne.regularization.l2)
	else:
		loss = loss.mean() 

	# create parameter update expressions
	params = lasagne.layers.get_all_params(network, trainable=True)
	#updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=opts['step_size'], momentum=opts['momentum'])
	updates = updates = lasagne.updates.sgd(loss, params, learning_rate=opts['step_size'])

	# use trained network for predictions
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
	test_loss = test_loss.mean()

	#Create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

	#Compile a function to perform the training step on a mini-batch update
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	# Compile a second function computing the validation loss
	val_fn = theano.function([input_var, target_var], [test_loss,test_acc])

	return network, train_fn, val_fn, input_var

# Function: train_network
# 
# Description: trains the neural network based on a training function, validation function, and num_epochs value
#
def train_network(network,train_fn,val_fn, X_train,y_train, X_val,y_val,opts):
	epoch = 0

	while epoch < opts['num_epochs']:
		train_err = 0
		train_batches = 0
		start_time = time.time()
	
		for batch in iterate_minibatches(X_train,y_train,opts['batch_size'],shuffle =True):
			inputs, targets = batch
			train_err += train_fn(inputs,targets)
			train_batches += 1

		val_err = 0
		val_acc = 0
		val_batches = 0

		for batch in iterate_minibatches(X_val, y_val,opts['batch_size'], shuffle = False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		if(not opts['mute_training_output']):
			print("Epoch {} of {} took {:.3f}s".format(
				epoch + 1, opts['num_epochs'], time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("  validation accuracy:\t\t{:.2f}%".format(
	            val_acc / val_batches * 100))

		epoch +=1

	return (val_err/val_batches)

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
#
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)

		yield inputs[excerpt], targets[excerpt]

#Function: get_predictions
#
#Description: Get predictions for input vectors given a network
#
#
def get_predictions(network,input_var, X_test):

	#Prediction functions
	prediction = lasagne.layers.get_output(network,deterministic=True)
	predict_function = theano.function([input_var], prediction)

	predictions = predict_function(X_test)

	print(predictions.shape)
	return predictions

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

#Function: get_neural_net_error
#
#Description: Setup and train a neural network given options and data
#
def get_neural_net_error(opts, X_train, y_train, X_val, y_val, df):

	# Prepare Theano variables
	target_var = T.ivector('target')

	#setup Network
	network, train_fn, val_fn, input_var = setup_network(len(df.columns[1:]), target_var, opts)

	#train network
	error = train_network(network,train_fn,val_fn,X_train, y_train, X_val, y_val, opts)

	return error


#Function: get_neural_net
#
#Description: Setup and train a neural network given options and data
#
def get_neural_net(opts, X_train, y_train, X_val, y_val, df):

	# Prepare Theano variables
	target_var = T.ivector('target')

	print("Setting up network...")
	#setup Network
	network, train_fn, val_fn, input_var = setup_network(len(df.columns[1:]), target_var, opts)

	print("Starting training...")
	#train network
	error = train_network(network,train_fn,val_fn,X_train, y_train, X_val, y_val, opts)

	if(error == None):
		print("NN has NAN error")
		print(opts)
		sys.exit(-1)

	return network, input_var


#Function: get_neural_net
#
#Description: Setup and train a neural network given options and data
#
def get_neural_net_scaled(K,X_train, y_train, X_val, y_val, df):
	#Set paramters
	opts = {
		'step_size' : .02,				#Step size for gradient updates
		'momentum' :.9,
		'num_epochs' : 20,				#Maximum number of epochs during training
		'batch_size' : 30,				#Batch size used during training
		'two_hidden_layers': 1,			#1 or 2 hidden layers
		'hidden_units_1': 30,			#Hidden units in layer 1
		'hidden_units_2': 10,			#Hidden units in layer 2
		'dropout_rate_1': .05, 			#Dropout rate between input and layer 1
		'dropout_rate_2': .05, 			#Dropout rate between layer 1 and layer 2
		'activation_func_1': 'sigmoid', #Activation function used in layer 1
		'activation_func_2': 'tanh',	#Activation function used in layer 2
		'mute_training_output': False,  #Show the stats on training on each iterations
	}


	# Prepare Theano variables
	target_var = T.ivector('target')

	print("Setting up network...")
	#setup Network
	network, train_fn, val_fn, input_var = setup_network(len(X_train[0,0,0:]), target_var, opts)

	print("Starting training...")
	#train network
	error = train_network(network,train_fn,val_fn,X_train, y_train, X_val, y_val, opts)

	if(error == None):
		print("NN has NAN error")
		print(opts)
		sys.exit(-1)

	return network, input_var


#Function: get_neural_net
#
#Description: Setup and train a neural network given options and data
#
def get_neural_net_default(X_train, y_train, X_val, y_val, df):
	
	#Set paramters
	opts = {
		'step_size' : .01,				#Step size for gradient updates
		'momentum' :.9,
		'num_epochs' : 20,				#Maximum number of epochs during training
		'batch_size' : 30,				#Batch size used during training
		'two_hidden_layers': False,			#1 or 2 hidden layers
		'hidden_units_1': 30,			#Hidden units in layer 1
		'hidden_units_2': 0,			#Hidden units in layer 2
		'dropout_rate_1': 0, 			#Dropout rate between input and layer 1
		'dropout_rate_2': 0, 			#Dropout rate between layer 1 and layer 2
		'activation_func_1': 'sigmoid', #Activation function used in layer 1
		'activation_func_2': 'tanh',	#Activation function used in layer 2
		'mute_training_output': False,  #Show the stats on training on each iterations
	}


	# Prepare Theano variables
	target_var = T.ivector('target')

	print("Setting up network...")
	#setup Network
	network, train_fn, val_fn, input_var = setup_network(len(X_train[0,0,0:]), target_var, opts)

	print("Starting training...")
	#train network
	error = train_network(network,train_fn,val_fn,X_train, y_train, X_val, y_val, opts)

	if(error == None):
		print("NN has NAN error")
		print(opts)
		sys.exit(-1)

	return network, input_var



# Main is based off of code from example of neural net from Lasagne distribution
# found at https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py#L213
#
if __name__ == "__main__":

	#Set paramters
	opts = {
		'step_size' : 1e-3,				#Step size for gradient updates .01
		'momentum' :.9,
		'num_epochs' : 50,				#Maximum number of epochs during training
		'batch_size' : 30,				#Batch size used during training
		#'alpha': 1e-4,
		'two_hidden_layers': False,			#1 or 2 hidden layers
		'hidden_units_1': 100,			#Hidden units in layer 1
		'hidden_units_2': 0,			#Hidden units in layer 2
		'dropout_rate_1': 0, 			#Dropout rate between input and layer 1
		'dropout_rate_2': 0, 			#Dropout rate between layer 1 and layer 2
		'activation_func_1': 'sigmoid', #Activation function used in layer 1
		'activation_func_2': 'sigmoid',	#Activation function used in layer 2
		'mute_training_output': False,  #Show the stats on training on each iterations
	}

	print("Loading training data...")
	#Load training data 			 
	Training_Data_File = sys.argv[1]
	X_train, y_train, X_val, y_val, df = get_training_data(Training_Data_File)

	network, input_var = get_neural_net(opts, X_train, y_train, X_val, y_val, df)

	print("Loading testing data")
	X_test,ids_test = get_test_data(sys.argv[2])

	print("Making predictions...")
	predictions = get_predictions(network,input_var, X_test)

	print("Writing predictions to file...")
	result_filename = sys.argv[3]
	write_results_file(result_filename,predictions,ids_test)

	print("Complete.")


