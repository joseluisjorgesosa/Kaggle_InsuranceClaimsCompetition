import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
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
def get_training_data(training_file,split): 
	ids_train, df = load_data(training_file)

	traindf, valdf = train_test_split(df, test_size = split)

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

	return X_train, y_train, X_val, y_val, valdf,traindf, df

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

	network = lasagne.layers.InputLayer(shape=(None,1,numFeatures),bias=lasagne.init.Constant(1))
	input_var = network.input_var

	network = lasagne.layers.BiasLayer(network, 
				b=lasagne.init.Constant(-10))

	network = lasagne.layers.DenseLayer(
				network, num_units=100,
				nonlinearity=lasagne.nonlinearities.sigmoid,
				W=lasagne.init.GlorotNormal(gain=1))

	network = lasagne.layers.DropoutLayer(network, p=0.2, rescale=True)

	network = lasagne.layers.DenseLayer(
				network, num_units=40,
				nonlinearity=lasagne.nonlinearities.ScaledTanH(scale_in=1, scale_out=1),
				W=lasagne.init.GlorotNormal(gain=1))

	network = lasagne.layers.DropoutLayer(network, p=0.15, rescale=True)

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
def setup_network(NUM_FEATURES, target_var, opts):

	input_var, network = build_mlp(NUM_FEATURES)

	#create loss function 
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()

	# create parameter update expressions
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=opts['step_size'], momentum=opts['momentum'])
	
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
	saved_validation_loss = np.empty(opts['num_epochs'])
	saved_training_loss = np.empty(opts['num_epochs'])
	
	epoch = 0
	prev_err = 10000000
	delta_err = 1000000

	while epoch < opts['num_epochs'] and delta_err > 1e-6:
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

		saved_validation_loss[epoch] = val_err
		saved_training_loss[epoch] = train_err
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, opts['num_epochs'], time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f}%".format(
            val_acc / val_batches * 100))

		delta_err = abs((train_err/train_batches) - (prev_err))
		prev_err = (train_err/train_batches)
		print("  delta err:\t\t{:.6f}".format(delta_err))

		epoch +=1

	t = np.asarray(range(opts['num_epochs'])) 
	s = saved_validation_loss
	r = saved_training_loss
	plt.plot(t, s,'r^',t,r,'b^')
	plt.xlabel('Epoch')
	plt.ylabel('Validation Error')
	plt.title('Validation Error Over Neural Net Training')
	plt.grid(True)
	plt.savefig("testNN.png")
	#plt.show()

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


# Main is based off of code from example of neural net from Lasagne distribution
# found at https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py#L213
#
if __name__ == "__main__":

	#Set paramters
	opts = {
		'step_size' : .001,			#Step size for gradient updates
		'momentum' :.9,
		'num_epochs' : 1,			#Maximum number of epochs during training
		'batch_size' : 300,		#Batch size used during training

	}

	print("Loading training data...")
	#Load training data 			 
	Training_Data_File = sys.argv[1]
	X_train, y_train, X_val, y_val,valdf, traindf, df = get_training_data(Training_Data_File, split=.5)

	# Prepare Theano variables
	target_var1 = T.ivector('target')
	target_var2 = T.ivector('target')

	print("Setting up network...")
	#setup Network
	NUM_FEATURES = len(traindf.columns[1:]) 
	network1, train_fn1, val_fn1, input_var1 = setup_network(NUM_FEATURES, target_var1, opts)
	network2, train_fn2, val_fn2, input_var2 = setup_network(NUM_FEATURES, target_var2, opts)

	print("Starting training...")
	#train network
	train_network(network1,train_fn1,val_fn1,X_train, y_train, X_val, y_val, opts)
	train_network(network2,train_fn2,val_fn2,X_val, y_val, X_train, y_train, opts)

	print("Making predictions...")
	predictions1 = get_predictions(network1,input_var1, X_val)
	predictions2 = get_predictions(network2,input_var2, X_train)


	#Combine labels again
	y_stack = np.zeros(len(df))
	y_stack[0:len(traindf)] = y_train[0:]
	y_stack[len(traindf):len(traindf)+len(valdf)] = y_val[0:]

	for i in range(len(y_stack)):
		print(y_stack[i])

	#Combine prediction columns
	X_stack = np.zeros((len(df),1,1))
	X_stack[0:len(traindf),0,0] = predictions2[0:,1]
	X_stack[len(traindf):len(traindf)+len(valdf),0,0] = predictions1[0:,1]

	all_stack = np.zeros((len(df),1,2))
	all_stack[0:len(df),0,0] = y_stack
	all_stack[0:len(df),0,1] = X_stack[0:,0,0]


	########################################################################
	# TODO: APPEND LABELS TO DATA, TRAIN ON IT 							#
	# THEN APPEND PREDICTIONS FOR X_TEST TO X_TEST AND TRAIN ON X_test  #
	########################################################################

	#Now complete stack trainig

	#Set paramters
	opts = {
		'step_size' : .0001,			#Step size for gradient updates
		'momentum' :.9,
		'num_epochs' : 4,			#Maximum number of epochs during training
		'batch_size' : 300,		#Batch size used during training

	}

	stack_train, stack_val = train_test_split(all_stack, test_size = 0.2)

	X_stack_val = np.zeros((len(stack_val),1,1))
	X_stack_val[0:len(stack_val),0,0] = stack_val[0:,0,1]
	X_stack_val = X_stack_val.astype(floatX)

	y_stack_val = stack_val[0:,0,1].astype('int32')

	X_stack_train = np.zeros((len(stack_train),1,1))
	X_stack_train[0:len(stack_train),0,0] = stack_train[0:,0,1]
	X_stack_train = X_stack_train.astype(floatX)

	y_stack_train = stack_train[0:,0,1].astype('int32')

	# Prepare Theano variables
	target_var_stack = T.ivector('target')

	print("Setting up stack network...")
	
	#setup Network 
	network_stack, train_fn_stack, val_fn_stack, input_var_stack = setup_network(1, target_var_stack, opts)

	print("Starting training...")
	#train network
	train_network(network_stack,train_fn_stack,val_fn_stack,X_stack_train, y_stack_train, X_stack_val, y_stack_val, opts)


	#Now complete full training
	#Set paramters
	opts = {
		'step_size' : .001,			#Step size for gradient updates
		'momentum' :.9,
		'num_epochs' : 20,			#Maximum number of epochs during training
		'batch_size' : 300,		#Batch size used during training

	}

	X_train, y_train, X_val, y_val,valdf, traindf, df = get_training_data(sys.argv[1], 0.2)

	# Prepare Theano variables
	target_var = T.ivector('target')

	print("Setting up network...")
	#setup Network
	network, train_fn, val_fn, input_var = setup_network(NUM_FEATURES, target_var, opts)

	print("Starting training...")
	#train network
	train_network(network,train_fn,val_fn,X_train, y_train, X_val, y_val, opts)

	print("Loading testing data")
	X_test,ids_test = get_test_data(sys.argv[2])

	print("Making predictions...")
	predictions = get_predictions(network,input_var, X_test)

	predictions_stack = np.zeros((len(predictions),1,1))
	predictions_stack[0:,0,0] = predictions[0:,1] 
	predictions_stack = predictions_stack.astype(floatX)

	predictions_stack = get_predictions(network_stack,input_var_stack,predictions_stack)

	print("Writing predictions to file...")
	result_filename = sys.argv[3]
	write_results_file(result_filename,predictions_stack,ids_test)

	print("Complete.")

