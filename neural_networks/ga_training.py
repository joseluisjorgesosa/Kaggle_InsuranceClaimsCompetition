import simpleNN as snn 
import random_parameters as rp
import operator 
import random
import sys

#Holds training data for the networks after it is loaded in. None of these ever change.
X_TRAIN = None
Y_TRAIN = None
X_VAL  = None
Y_VAL = None
DF = None

#Function: load_data
#
#Description: loads trianing data into global vars
#
#
def load_data(filename):
	return  snn.get_training_data(filename)

#Function: evolve_population
#
#Description: Evolves population iters number of times and then returns the new population
#
#

def evolve_population(population,max_iters):
	#Evolve population max iters number of times
	for j in range(max_iters):
		print(str(j) + " of " + str(max_iters) + " iterations")
		new_pop = []
		for i in range(len(population)):
			net_a = get_random_network(population)
			net_b = get_random_network(population)

			new_net = reproduce(net_a,net_b)

			new_net = mutate(new_net)

			new_pop.insert(i,new_net)
			print("\t"+ str(i) + " of " + str(len(population)) + " individuals produced")

		new_pop = population

	return population


#Function get_random_network()
#
#Description: Returns a random network, with networks with lower training error having a higher 
#			 probability of being selected
#
def get_random_network(population):
	p = .25

	sorted_nets =  sorted(population, key=operator.itemgetter(1))
	network = []

	for i in range(len(sorted_nets)):
		network = sorted_nets[i]

		if random.random()>p:
			break

	return network

#Function get_top_networks
#
#Description: returns the parameters of the best neural networks
#
def get_top_networks(num_nets, population):

	sorted_nets =  sorted(population, key=operator.itemgetter(1))

	return sorted_nets[0:min(len(sorted_nets),num_nets)]

#Description: Given two networks, creates a new network that is a combination of the previous two
#
def reproduce(net_a,net_b):

	#Get options from input nets
	opts_a = net_a[0]
	opts_b = net_b[0]

	#Create empty options
	network = []

	#Get options for network so they can be added
	opts = rp.get_default_opts()

	opts['hidden_units_1'] = rp.pick_random(opts_a['hidden_units_1'],opts_b['hidden_units_1'])
	opts['hidden_units_2'] = rp.pick_random(opts_a['hidden_units_2'],opts_b['hidden_units_2'])

	opts['dropout_rate_1'] = rp.pick_random(opts_a['dropout_rate_1'],opts_b['dropout_rate_1'])
	opts['dropout_rate_2'] = rp.pick_random(opts_a['dropout_rate_2'],opts_b['dropout_rate_2'])

	opts['activation_func_1'] = rp.pick_random(opts_a['activation_func_1'],opts_b['activation_func_1'])
	opts['activation_func_2'] = rp.pick_random(opts_a['activation_func_2'],opts_b['activation_func_2'])

	network.insert(0,opts)
	network.insert(1,snn.get_neural_net_error(opts, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, DF))

	return network 

#Function: Mutate
#
#Description: Mutates a network randomly (with certain probability returns a network that hasn't been updated)
#
def mutate(net):
	opts = net[0]

	x = int(round(random.random()*4))

	if(x == 0):
		opts['hidden_units_1'] = opts['hidden_units_1']*(2-2*random.random())
	elif(x == 1):
		opts['hidden_units_2'] = opts['hidden_units_2']*(2-2*random.random())
	elif(x == 2):
		opts['dropout_rate_1'] = opts['dropout_rate_1']*(2-2*random.random())
	elif(x == 3):
		opts['dropout_rate_2'] = opts['dropout_rate_2']*(2-2*random.random())


	net[0] = opts

	return net



# Function: initialize_population
#
# Description: Creates an initial population of networks to be trained
#
def initialize_population(init_popSize):
	population = []


	#Initialize a population of neural nets
	for i in range(init_popSize):
		network = []
		network.insert(0, rp.get_random_opts()),
		network.insert(1,i,),	
			
		population.append(network)

	return population

def write_to_file(nets, results_filename):
		
		results = open(results_filename,'a')
 	
 		for network in nets:
 			opts = network[0]

 			net_info = " "
 			net_info = net_info + str(opts['step_size'])
 			net_info = net_info +","+ str(opts['momentum'])
 			net_info = net_info +","+ str(opts['num_epochs'])
 			net_info = net_info +","+ str(opts['batch_size'])
 			net_info = net_info +","+ str(opts['hidden_units_1'])
 			net_info = net_info +","+ str(opts['hidden_units_2'])
			net_info = net_info +","+ str(opts['dropout_rate_1'])
			net_info = net_info +","+ str(opts['dropout_rate_2'])
			net_info = net_info +","+ str(opts['activation_func_1'])
			net_info = net_info +","+ str(opts['activation_func_2']) + "\n"

			print(net_info)

			results.write(net_info)
	
		results.close();

# Main: Genetic Algorithm for Parameter Tuning of Neural Networks
#
#
#
if __name__ == "__main__":

	#Initial Pop Size
	init_popSize = 20

	#Maximum iterations
	max_iters = 5

	print("Loading data...")
	X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, DF = load_data(sys.argv[1])


	print("Initializing population...")
	population = initialize_population(init_popSize)


	print("Starting Evolution...")
	population = evolve_population(population,max_iters)

	print("Getting top neural nets..")
	selected_neural_nets = get_top_networks(10,population)

	print("Writing to file...")
	write_to_file(selected_neural_nets,sys.argv[2])





