#!/usr/bin/python

'''Top layer for generating predictions. Ensembles 3 different models (Neural Nets, XGBoost, Random Forests) using XGBoost.'''
__author__ = 'Orestis Lykouropoulos'

#import built in modules here
import simplexgb as x
import ensemble as nn
import extra_trees_func as rf

import sys
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


def merge_first_layer_res(nn_res, xgb_res, rf_res):
	'''
	Merge first layer results (probabilities) into a Data Frame with 3 columns, one for each model's results

	Inputs: nn_res:  3-tuple of train, validation and test probabilities from Neural Nets
			xgb_res: 3-tuple of train, validation and test probabilities from XGBoost (first layer)
			rf_res:  3-tuple of train, validation and test probabilities from Random Forests

	Output: 3-tuple of merged Train, Validation and Test Data Frames
	'''

	columns = ['NN', 'XGB', 'RF']
	train_df = get_merged_df([nn_res[0], xgb_res[0], rf_res[0]], columns)
	validation_df = get_merged_df([nn_res[1], xgb_res[1], rf_res[1]], columns)
	test_df = get_merged_df([nn_res[2], xgb_res[2], rf_res[2]], columns)

	return train_df, validation_df, test_df

def get_merged_df(np_list, labels):
	df_list = [None] * len(np_list)
	for i in range(len(np_list)):
		df_list[i] = pd.DataFrame(np_list[i], columns = [labels[i]])
	return pd.concat(df_list, axis=1)

def write_results(preds, ids_test):
	'''
	Write results csv file. Each line is ID,probablity pair
	'''
	submission = open('top_layer_results.csv','w')
	submission.write('ID,PredictedProb\n')

	for i in range(len(preds)):
		submission.write(str(ids_test[i]) + "," + str(preds[i])+"\n")
	submission.close()

def run_top_xgb(nn_res, xgb_res, rf_res, train_labels, validation_labels, ids_test):
	'''
	Train XGBoost model on train_df of probabilities generated by the 3 first layer models. Monitor performance with validation_df.
	Return predictions on test_df
	'''

	train_df, validation_df, test_df = merge_first_layer_res(nn_res, xgb_res, rf_res)

	train_DMatrix = xgb.DMatrix(train_df, train_labels)
	validation_DMatrix = xgb.DMatrix(validation_df, validation_labels) #for validation/early stopping

	param = {'max_depth':10, 'eta':0.1, 'min_child_weight':1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', 'subsample':1, 'col_sample_bytree':0.8 }

	#specify validation set to watch performance
	watchlist  = [(train_DMatrix,'train'), (validation_DMatrix,'eval')]
	num_round = 2000
	
	bst = xgb.train(param, train_DMatrix, num_round, watchlist, early_stopping_rounds=100)

	#prediction
	test_DMatrix = xgb.DMatrix(test_df)
	preds = bst.predict(test_DMatrix, ntree_limit=bst.best_ntree_limit)
	write_results(preds, ids_test) #need to get ids from somewhere
	return preds

if __name__ == '__main__':

	# nn = [[] for i in range(3)]
	# nn[0] = np.random.rand(10)
	# nn[1] = np.random.rand(10)
	# nn[2] = np.random.rand(10)
	# train_labels = [int(round(i)) for i in nn[0]]
	# validation_labels = [int(round(i)) for i in nn[1]]
	# print train_labels
	# print validation_labels
	# print nn

	# x = [[] for i in range(3)]
	# x[0] = nn[0]
	# x[1] = nn[1]
	# x[2] = nn[2]
	# print x

	# rf = [[] for i in range(3)]
	# rf[0] = np.random.rand(10)
	# rf[1] = np.random.rand(10)
	# rf[2] = np.random.rand(10)
	# print rf


	# preds = run_top_xgb(nn, x, rf, train_labels, validation_labels, [1,2,3,4,5,7,8,9,24,80])
	# print preds
	# print nn[2]
	# print x[2]
	# print rf[2]	

	train_file = 'train.csv'
	test_file = 'test.csv' 

	nn_res = nn.get_predictions(train_file,test_file) #bailey
	rf_res = rf.get_predictions(train_file, test_file) #jojo
	train_probs, validation_probs, test_probs, train_labels, validation_labels, ids_test = x.run_first_layer_xgb(train_file, test_file) #orestis
	x_res = train_probs, validation_probs, test_probs
	print "run top\n\n\n\n"
	

	run_top_xgb(nn_res, rf_res, x_res, train_labels, validation_labels, ids_test)







