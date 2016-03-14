import pandas as pd

def average_preds(flist):
	'''
	Load the results list and return averaged results
	'''
	dflist = [None] * len(flist)
	for i in range(len(flist)):
		dflist[i] = pd.read_csv(flist[i])
	
	dftotal = dflist[0]
	for df in dflist[1:]:
		dftotal = dftotal.add(df)

	return dftotal.multiply(1.0/len(flist))

def get_ids(test_data):
	df = pd.read_csv(test_data)
	return df['ID']

def write_results(preds, ids_test):
	'''
	Write results csv file. Each line is ID,probablity pair
	'''
	submission = open('averaged_results.csv','w')
	submission.write('ID,PredictedProb\n')

	for i in range(len(preds)):
		submission.write(str(ids_test[i]) + "," + str(preds[i])+"\n")
	submission.close()

if __name__ == '__main__':
	preds = average_preds(['sample_submission.csv', 'sample_submission.csv', 'sample_submission.csv'])
	ids = get_ids('sample_submission.csv')
	write_results(preds['PredictedProb'], ids)