#!/usr/bin/python
import sys
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


def load_data(data_file):
	'''
	Load the training and testing data
	'''
	df = pd.read_csv(data_file)

	#separate id column
	id_col = df['ID']
	cols = df.columns.tolist()[1:]
	df = df[cols]

	#replace missing values with large negative value (to encode the absence of the variable)
	df.fillna(-9999, inplace=True)

	#encode categorical features as integers
	cat_columns = df.select_dtypes(['object']).columns
	for c in cat_columns:
		df[c] = df[c].astype('category')
	df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

	return id_col, df

def write_results(pred, ids_test):
	'''
	Write results csv file. Each line is ID,probablity pair
	'''
	submission = open('results.csv','w')
	submission.write('ID,PredictedProb\n')

	for i in range(len(preds)):
		submission.write(str(ids_test[i]) + "," + str(preds[i])+"\n")
	submission.close()

if __name__=="__main__":

	if len(sys.argv) != 3:
		print("Usage: python simplexgb.py training_csv_file, testing_csv_file")
		sys.exit(1)

	#load data
	_, df = load_data(sys.argv[1])
	ids_test, predict_df = load_data(sys.argv[2]) #data set to predict values for

	#split training set 80/20 and train on 80% use 20% to evaluate
	traindf, evaldf = train_test_split(df, test_size = 0.2)

	features = list(traindf.columns[1:])
	labels = traindf.columns[0]
	train_DMatrix = xgb.DMatrix(traindf[features], traindf[labels])
	eval_DMatrix = xgb.DMatrix(evaldf[features], evaldf[labels]) #for validation/early stopping

	#specify parameters
	param = {'max_depth':6, 'eta':0.1, 'subsample':1, 'min_child_weight':5, 
	'col_sample_bytree':0.8, 'silent':1, 'objective':'reg:linear'}

	#specify validation set to watch performance
	watchlist  = [(train_DMatrix,'train'), (eval_DMatrix,'eval')]
	num_round = 200
	bst = xgb.train(param, train_DMatrix, num_round, watchlist, early_stopping_rounds=10)

	#prediction
	predict_DMatrix = xgb.DMatrix(predict_df)
	preds = bst.predict(predict_DMatrix)
	write_results(preds, ids_test)

	# bst.save_model('0001.model')