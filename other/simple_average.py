import pandas as pd

def average_preds(f1, f2, f3):
	'''
	Load the 3 results
	'''
	df1 = pd.read_csv(f1)
	df2 = pd.read_csv(f2)
	df3 = pd.read_csv(f3)

	df_med = df1.add(df2)
	df_final = df_med.add(df3)

	return df_final.multiply(1/3.0)

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
	preds = average_preds('orestis.csv', 'bailey.csv', 'jojo.csv')
	ids = get_ids('orestis.csv')
	write_results(preds['PredictedProb'], ids)