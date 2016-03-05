import sys
import pandas as pd
from sklearn import preprocessing
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
def get_training_data(training_file): 
	ids_train, df = load_data(training_file)

	traindf, valdf = train_test_split(df, test_size = 0.2)

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

def build_mlp(numFeatures):

	network = lasagne.layers.InputLayer(shape=(None,1,numFeatures))

	input_var = network.input_var

	network = lasagne.layers.ScaleLayer(network, 
				scales=lasagne.init.Constant(1))

	network = lasagne.layers.DenseLayer(
				network, num_units=500,
				nonlinearity=lasagne.nonlinearities.rectify,
				W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.))

	network = lasagne.layers.BiasLayer(network, 
				b=lasagne.init.Constant(0))

	network = lasagne.layers.DenseLayer(
				network, num_units=500,
				nonlinearity=lasagne.nonlinearities.sigmoid,
				W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.))

	network = lasagne.layers.NonlinearityLayer(network, 
		nonlinearity=lasagne.nonlinearities.rectify)

	network = lasagne.layers.DenseLayer(
		network, num_units=2,
		nonlinearity=lasagne.nonlinearities.softmax)

	return input_var, network

# Function: build_custom_mlp	
# Description: Builds a custom mlp based on the Lasagne documentation
#
# Creates a neural net with the following specified by a user:
#	
#		-number of input features 
#		-number of hidden layers
#		-hidden units
#		-input dropout (boolean)
#		-hidden dropout (boolean)

def build_custom_mlp(numFeatures,  drop_input = False, hidden_layers=1, hidden_units=800, drop_hidden=False, bias=False):
	
	network = lasagne.layers.InputLayer(shape=(None,1,numFeatures))
	
	input_var = network.input_var

	if drop_input:
		network = lasagne.layers.DropoutLayer(network, p=0.2)



	for i in range(hidden_layers):
		if bias:
			network = lasagne.layers.DenseLayer(
				network, num_units=hidden_units,
				nonlinearity=lasagne.nonlinearities.rectify,
				W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.))
		else:
			network = lasagne.layers.DenseLayer(
				network, num_units=hidden_units,
				nonlinearity=lasagne.nonlinearities.rectify,
				W=lasagne.init.GlorotUniform())

		if drop_hidden:
			network = lasagne.layers.DropoutLayer(network, p=0.5)


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
def setup_network(df, target_var, opts):

	#Build custom netowrk
	if(opts['customNet'] == True):
		input_var, network = build_custom_mlp(len(df.columns[1:]), opts['drop_input'], opts['hidden_layers'], opts['hidden_units'], opts['drop_hidden'],opts['bias'])
	else:
		input_var, network = build_mlp(len(df.columns[1:]))

	#create loss function 
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

	if(opts['L2Regularization'] == False):
		loss = loss.mean()
	else:
		loss = loss.mean()+ opts['alpha'] * lasagne.regularization.regularize_network_params( network, lasagne.regularization.l2)

	# create parameter update expressions
	params = lasagne.layers.get_all_params(network, trainable=True)

	if(opts['updateFunction'] == 'SGD'):
		updates = lasagne.updates.sgd(loss, params, learning_rate=opts['step_size'])

	elif(opts['updateFunction'] == 'Adagrad'):
		updates = lasagne.updates.adagrad(loss, params, learning_rate=opts['step_size'], epsilon=opts['epsilon'])

	elif(opts['updateFunction'] == 'Adamax'):
		#Recomended step size = .002, beta1 =.9, beta2=.999, epsilon = 1e-8
		updates = lasagne.updates.adamax(loss, params, learning_rate=opts['step_size'], beta1=opts['beta1'], beta2=opts['beta2'], epsilon=opts['epsilon'])
	
	elif(opts['updateFunction'] == 'Adadelta'):
		updates = lasagne.updates.adadelta(loss, params, learning_rate=opts['step_size'],  rho=opts['rho'], epsilon=opts['epsilon']) 
	
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

	for epoch in range(opts['num_epochs']):
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

		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, opts['num_epochs'], time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


	

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
		results.write(str(ids_test[i]) + "," + str(predictions[i]) + "\n")

	results.close()

# Main is based off of code from example of neural net from Lasagne distribution
# found at https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py#L213
#
if __name__ == "__main__":

	#Set paramters
	opts = {
		'net1': False,				#If true uses a neural net that is predetermined
		'customNet': True,			#If true uses a neural net specified by parameters below
		'updateFunction':'Adamax',	#Select the update function (SGD, Adagrad, Adamax)
		'L2Regularization': True,	#Determine if Regularization is used or not
		'alpha' :  1e-3,			#L2 regularization scalar
		'rho' : 1e-6,				#Rho (Adadelta)
		'beta1' : .9,				#Beta 1 (Adamax)
		'beta2' : .999,				#Beta 2 (Adamax)
		'step_size' : 2e-3,			#Step size for gradient updates
		'epsilon' : 1e-8,			#Used to control how quickly step size decreases for Adagrad/Adamax
		'num_epochs' : 10,			#Maximum number of epochs during training
		'batch_size' : 10,			#Batch size used during training
		'hidden_layers' : 10,		#Total number of hidden layers
		'hidden_units': 200,		#Hidden units in each layer
		'drop_input' : False,		#Randomly remove a certain proportion of input units
		'drop_hidden': True,		#Randomly remove a certain proportion of hidden units
		'bias': True,				#Add bias into hidden layers
	}

	print("Loading training data...")
	#Load training data 			 
	Training_Data_File = sys.argv[1]
	X_train, y_train, X_val, y_val, df = get_training_data(Training_Data_File)


	# Prepare Theano variables
	target_var = T.ivector('targets')

	print("Setting up network...")
	#setup Network
	network, train_fn, val_fn, input_var = setup_network(df, target_var, opts)

	print("Starting training...")
	#train network
	train_network(network,train_fn,val_fn,X_train, y_train, X_val, y_val, opts)

	print("Saving network parameters")
	#Save trained network 
	np.savez('model.npz', *lasagne.layers.get_all_param_values(network))

	print("Loading testing data")
	X_test,ids_test = get_test_data(sys.argv[2])

	print("Making predictions...")
	predictions = get_predictions(network,input_var, X_test)

	print("Writing predictions to file...")
	result_filename = sys.argv[3]
	write_results_file(result_filename,predictions,ids_test)

	print("Complete.")


