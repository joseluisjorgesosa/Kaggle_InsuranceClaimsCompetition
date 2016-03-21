#!/usr/bin/python

'''XGBoost first layer model and prediction generation.'''
__author__ = 'Orestis Lykouropoulos'

import sys
import xgboost as xgb
import pandas as pd
import numpy
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
	df.drop('v91', axis=1, inplace=True)

	return id_col, df

def run_first_layer_xgb(training_csv_file, testing_csv_file, seed = 1, validation_size = 0.20):

	print('\nStarting first layer XGBoost\n')
	#load data
	print('Loading data...')
	_, df = load_data(training_csv_file)
	ids_test, predict_df = load_data(testing_csv_file) #data set to predict values for

	print('Splitting data for training and validation...')
	#split training set 80/20 and train on 80% use 20% to evaluate
	traindf, evaldf = train_test_split(df, test_size = 0.20, random_state = seed)

	features = list(traindf.columns[1:])
	labels = traindf.columns[0]
	train_DMatrix = xgb.DMatrix(traindf[features], traindf[labels])
	eval_DMatrix = xgb.DMatrix(evaldf[features], evaldf[labels]) #for validation/early stopping

	#specify validation set to watch performance
	watchlist  = [(train_DMatrix,'train'), (eval_DMatrix,'eval')]
	num_round = 70
	param = {'max_depth':6, 'eta':0.1, 'min_child_weight':5, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', 'subsample':.7, 'col_sample_bytree':0.8 }

	print('Training. Don\'t hold your breath...')
	bst = xgb.train(param, train_DMatrix, num_round, watchlist)#, early_stopping_rounds=20)

	print('Calculating predictions based on model...')
	#predictions on training data
	train_probs = bst.predict(train_DMatrix)#, ntree_limit=bst.best_ntree_limit)

	#predictions on validation data
	validation_probs = bst.predict(eval_DMatrix)#, ntree_limit=bst.best_ntree_limit)

	#predictions on test data
	predict_DMatrix = xgb.DMatrix(predict_df)
	test_probs = bst.predict(predict_DMatrix)#, ntree_limit=bst.best_ntree_limit)

	print('Done')
	return train_probs, validation_probs,  test_probs, traindf[labels], evaldf[labels], ids_test

# *********** No longer used functions, used in calibration and model development, included for completeness. *****************************************
# *****************************************************************************************************************************************************

def write_results(preds, ids_test):
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

def average_preds(dflist):
	'''
	Return averaged results
	'''
	dftotal = dflist[0]
	for df in dflist[1:]:
		dftotal = numpy.add(dftotal, df)
	return numpy.multiply(dftotal, 1.0/len(dflist))

if __name__=="__main__":

	res = run_first_layer_xgb('train.csv', 'test.csv')
	print res[0][0:10]
	print res[3][0:10]

	# if len(sys.argv) != 3:
	# 	print("Usage: python simplexgb.py training_csv_file, testing_csv_file")
	# 	sys.exit(1)

	# #load data
	# _, df = load_data(sys.argv[1])
	# ids_test, predict_df = load_data(sys.argv[2]) #data set to predict values for

	# #split training set 80/20 and train on 80% use 20% to evaluate
	# traindf, evaldf = train_test_split(df, test_size = 0.20, random_state=1)

	# features = list(traindf.columns[1:])
	# labels = traindf.columns[0]
	# train_DMatrix = xgb.DMatrix(traindf[features], traindf[labels])
	# eval_DMatrix = xgb.DMatrix(evaldf[features], evaldf[labels]) #for validation/early stopping

	# #specify validation set to watch performance
	# watchlist  = [(train_DMatrix,'train'), (eval_DMatrix,'eval')]
	# num_round = 2000

	# # iterations = 1 #repeat this many times and get average at the end
	# # depths = [10] #optimal depths
	# # weights = [1]
	# # preds_list = [None] * iterations
	# # for i in range(iterations):
	# # 	#specify parameters
	# # 	# current optimal 6, 0.1, 5, 0.7, 0.8
	# # 	#or maybe 6, 0.1, 1, 0.5, 0.4
	# # 	#latest 1200 - 10, 0.01 1 1 0.8 0.8
	# # 	param = {'max_depth':depths[i], 'eta':0.1, 'min_child_weight':weights[i], 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', 'subsample':1, 'col_sample_bytree':0.8 }

	# # 	bst = xgb.train(param, train_DMatrix, num_round, watchlist, early_stopping_rounds=10)

	# # 	#prediction
	# # 	predict_DMatrix = xgb.DMatrix(predict_df)
	# # 	preds_list[i] = bst.predict(predict_DMatrix, ntree_limit=bst.best_ntree_limit)

	# param = {'max_depth':10, 'eta':0.01, 'min_child_weight':1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', 'subsample':1, 'col_sample_bytree':0.8 }

	# bst = xgb.train(param, train_DMatrix, num_round, watchlist, early_stopping_rounds=10)

	# #prediction
	# predict_DMatrix = xgb.DMatrix(predict_df)
	# preds = bst.predict(predict_DMatrix, ntree_limit=bst.best_ntree_limit)

	# write_results(preds, ids_test)

