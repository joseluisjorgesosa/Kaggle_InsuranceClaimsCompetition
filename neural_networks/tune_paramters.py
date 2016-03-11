import sys
import simpleNN as snn



def assign_hiddenUnits(opts, X_train, y_train, X_val, y_val, df, results):

	hiddenUnits = [10,50,100,200,500]

	for hu1 in range(len(hiddenUnits)):
		for hu2 in range(len(hiddenUnits)):
			opts['hidden_units_1'] = hiddenUnits[hu1]
			opts['hidden_units_2'] = hiddenUnits[hu2]

			assign_dropoutRate(opts, X_train, y_train, X_val, y_val, df, results)


def assign_dropoutRate(opts, X_train, y_train, X_val, y_val, df, results):

	dropoutRate = [.05, .1, .25, .5]

	for dr1 in range(len(dropoutRate)):
		for dr2 in range(len(dropoutRate)):
			opts['dropout_rate_1'] = dropoutRate[dr1]
			opts['dropout_rate_2'] = dropoutRate[dr2]

			assign_activation_function(opts, X_train, y_train, X_val, y_val, df, results)

def assign_activation_function(opts, X_train, y_train, X_val, y_val, df, results):
	
	activationFun = ['sigmoid'] #,'tanh','linear','rectify']

	for af1 in range(len(activationFun)):
		for af2 in range(len(activationFun)):
			opts['activation_func_1'] = activationFun[af1]
			opts['activation_func_2'] = activationFun[af2]

			assign_step_size(opts, X_train, y_train, X_val, y_val, df, results)

def assign_step_size(opts, X_train, y_train, X_val, y_val, df, results):
	
	step_size = [1] #, #[.01] # .001, .0001]

	for stp in range(len(step_size)):
		opts['step_size'] = step_size[stp]

		assign_momentum(opts, X_train, y_train, X_val, y_val, df, results)

def assign_momentum(opts, X_train, y_train, X_val, y_val, df, results):
	
	momentum = [.01] #9] #, .3] # .01]

	for m in range(len(momentum)):
		opts['momentum'] = momentum[m]

		assign_batch_size(opts, X_train, y_train, X_val, y_val, df, results)

def assign_batch_size(opts, X_train, y_train, X_val, y_val, df, results):

	batch_size = [30] # 100, 250]

	for b in range(len(batch_size)):
		opts['batch_size'] = batch_size[b]

		test_neural_net(opts, X_train, y_train, X_val, y_val, df, results)

def test_neural_net(opts, X_train, y_train, X_val, y_val, df,results_filename):
		
		results = open(results_filename,'a')

		error = snn.get_neural_net_error(opts, X_train, y_train, X_val, y_val)

		results.write("==================================================================================\n")
		results.write("Error:" + str(error)+ "\n")
		results.write("Step Size:" + str(opts['step_size']) + "\n")
		results.write("Momentum: " + str(opts['momentum']) + "\n")
		results.write("Number of Epochs: " + str(opts['num_epochs']) + "\n")
		results.write("Batch Size: " + str(opts['batch_size']) + "\n")
		results.write("Hidden Units 1: " + str(opts['hidden_units_1']) + "\n")
		results.write("Hidden Units 2: " + str(opts['hidden_units_2']) + "\n")
		results.write("Dropout Rate 1: " + str(opts['dropout_rate_1']) + "\n")
		results.write("Dropout Rate 2: " + str(opts['dropout_rate_2']) + "\n")
		results.write("Activation Func 1: " + str(opts['activation_func_1']) + "\n")
		results.write("Activation Func 2: " + str(opts['activation_func_2']) + "\n")
		results.write("==================================================================================\n")

		results.close();

if __name__ == "__main__":

	print("Loading training data...")
	#Load training data 			 
	Training_Data_File = sys.argv[1]
	X_train, y_train, X_val, y_val, df = snn.get_training_data(Training_Data_File)

	results = sys.argv[2]
										
	#Set paramters
	opts = {		
		'step_size' : .1,							#Step size for gradient updates
		'momentum' : .9,
		'num_epochs' : 20,							#Maximum number of epochs during training
		'batch_size' : 200,							#Batch size used during training
		'hidden_units_1': 10,						#Hidden units in layer 1
		'hiddden_units_2': 10,						#Hidden units in layer 2
		'dropout_rate_1': .1, 						#Dropout rate between input and layer 1
		'dropout_rate_2': .1, 						#Dropout rate between layer 1 and layer 2
		'activation_func_1': 'sigmoid', 			#Activation function used in layer 1
		'activation_func_2': 'sigmoid',				#Activation function used in layer 2
		'mute_training_output': True,  #			Show the stats on training on each iterations
	}

	assign_hiddenUnits(opts, X_train, y_train, X_val, y_val, df, results)


										


