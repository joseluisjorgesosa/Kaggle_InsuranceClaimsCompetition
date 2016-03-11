import sys
import simpleNN as snn
import random


def assign_hiddenUnits(opts):

	hiddenUnits = [10,50,100,200,500]

	opts['hidden_units_1'] = hiddenUnits[int(round(random.random()*(len(hiddenUnits)-1)))]
	opts['hidden_units_2'] = hiddenUnits[int(round(random.random()*(len(hiddenUnits)-1)))]

	return opts


def assign_dropoutRate(opts):

	dropoutRate = [.05, .1, .25, .5]

	opts['dropout_rate_1'] = dropoutRate[int(round(random.random()*(len(dropoutRate)-1)))]
	opts['dropout_rate_2'] = dropoutRate[int(round(random.random()*(len(dropoutRate)-1)))]


	return opts

def assign_activation_function(opts):
	
	activationFun = ['sigmoid','tanh','linear','rectify']

	opts['activation_func_1'] = activationFun[int(round(random.random()*(len(activationFun)-1)))]
	
	opts['activation_func_2'] = activationFun[int(round(random.random()*(len(activationFun)-1)))]


	return opts

#Function: pick_random
#
#Description: Picks a or b randomly (equal probability of selecting each )
#
def pick_random(a,b):

	if(random.random()>.5):
		return a
	else:
		return b

def get_default_opts():

	#Set paramters
	opts = {		
		'step_size' : .1,							#Step size for gradient updates
		'momentum' : .9,
		'num_epochs' : 20,							#Maximum number of epochs during training
		'batch_size' : 30,							#Batch size used during training
		'two_hidden_layers': True,						#Hidden units in layer 1
		'hidden_units_1': 10,						#Hidden units in layer 1
		'hiddden_units_2': 10,						#Hidden units in layer 2
		'dropout_rate_1': .1, 						#Dropout rate between input and layer 1
		'dropout_rate_2': .1, 						#Dropout rate between layer 1 and layer 2
		'activation_func_1': 'sigmoid', 			#Activation function used in layer 1
		'activation_func_2': 'sigmoid',				#Activation function used in layer 2
		'mute_training_output': True,  #			Show the stats on training on each iterations
	}

	return opts
	
def get_random_opts():
	
	#Set paramters
	opts = {		
		'step_size' : .1,							#Step size for gradient updates
		'momentum' : .9,
		'num_epochs' : 20,							#Maximum number of epochs during training
		'batch_size' : 30,							#Batch size used during training
		'two_hidden_layers': True,						#Hidden units in layer 1
		'hidden_units_1': 10,						#Hidden units in layer 1
		'hiddden_units_2': 10,						#Hidden units in layer 2
		'dropout_rate_1': .1, 						#Dropout rate between input and layer 1
		'dropout_rate_2': .1, 						#Dropout rate between layer 1 and layer 2
		'activation_func_1': 'sigmoid', 			#Activation function used in layer 1
		'activation_func_2': 'sigmoid',				#Activation function used in layer 2
		'mute_training_output': True,  #			Show the stats on training on each iterations
	}

	opts = assign_hiddenUnits(opts)
	opts = assign_dropoutRate(opts)
	opts = assign_activation_function(opts)


	return opts
										


