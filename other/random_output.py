import sys
import pandas
import random

def readTestData(test_data_filename):
		test_data = pandas.read_csv(test_data_filename)
		return test_data

def get_baseline(test_data):
	test_output = []

	for ID in  test_data.ID:
		test_pair = [ID,round(random.random(),2)]
		test_output.append(test_pair)

	return test_output

def create_submission(test_output):
	test_submission = open('results.csv','w')
	test_submission.write('ID,PredictedProb\n')

	for test_val in test_output:
		test_submission.write(str(test_val[0]) + "," + str(test_val[1])+"\n")

	test_submission.close()

if __name__ == "__main__":

	test_data_filename = sys.argv[1]

	test_data = readTestData(test_data_filename)

	test_output = get_baseline(test_data)

	create_submission(test_output)
