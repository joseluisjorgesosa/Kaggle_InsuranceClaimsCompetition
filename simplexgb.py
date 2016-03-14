#!/usr/bin/python
import sys
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction import DictVectorizer



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
	df.fillna(-999, inplace=True)

	#categorical columns
	cat_columns = df.select_dtypes(['object']).columns

	#normalize and convert categoricals to numbers
	for c in df.columns.tolist():
		if c == 'target':
			continue
		if c in cat_columns:
			df[c] = df[c].astype('category')
			continue
		df[c] = (df[c] - df[c].mean()) / (df[c].max() - df[c].min())		
	df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

	#delete unnecessary columns (see post https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19240/analysis-of-duplicate-variables-correlated-variables-large-post)
	# df.drop('v91', axis=1, inplace=True)



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

def write_stats(fname, col, best_score, best_iteration):
	stats = open(fname, 'a')
	stats.write('Run with max_depth = {0}, score: {1}, iteration: {2}\n'.format(col, best_score, best_iteration))
	stats.close()

if __name__=="__main__":

	if len(sys.argv) != 3:
		print("Usage: python simplexgb.py training_csv_file, testing_csv_file")
		sys.exit(1)

	#load data
	_, df = load_data(sys.argv[1])
	ids_test, predict_df = load_data(sys.argv[2]) #data set to predict values for

	#split training set 80/20 and train on 80% use 20% to evaluate
	traindf, evaldf = train_test_split(df, test_size = 0.20)

	features = list(traindf.columns[1:])
	labels = traindf.columns[0]
	train_DMatrix = xgb.DMatrix(traindf[features], traindf[labels])
	eval_DMatrix = xgb.DMatrix(evaldf[features], evaldf[labels]) #for validation/early stopping

	#specify parameters
	# current optimal 6, 0.1, 5, 0.7, 0.8
	#or maybe 6, 0.1, 1, 0.5, 0.4
	param = {'max_depth':10, 'eta':0.01, 'min_child_weight':1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', 'subsample':0.8, 'col_sample_bytree':0.8 }

	#specify validation set to watch performance
	watchlist  = [(train_DMatrix,'train'), (eval_DMatrix,'eval')]
	num_round = 1200
	bst = xgb.train(param, train_DMatrix, num_round, watchlist)#, early_stopping_rounds=10)

	#prediction
	predict_DMatrix = xgb.DMatrix(predict_df)
	preds = bst.predict(predict_DMatrix)
	write_results(preds, ids_test)

	# write_stats('xgb_stats', test, bst.best_score, bst.best_iteration)

	# bst.save_model('0001.model')