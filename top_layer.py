#!/usr/bin/python
import sys
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

#import models

'''
Top layer for generating predicitions. Ensembles 3 different models using XGBoost. (more info in diagram later)
'''

def write_results(pred, ids_test):
	'''
	Write results csv file. Each line is ID,probablity pair
	'''
	submission = open('top_layer_results.csv','w')
	submission.write('ID,PredictedProb\n')

	for i in range(len(preds)):
		submission.write(str(ids_test[i]) + "," + str(preds[i])+"\n")
	submission.close()

def run_top_xgb(train_df, validation_df, test_df):
	'''
	Train XGBoost model on train_df of probabilities generated by the 3 first layer models. Monitor performance with validation_df.
	Return predictions on test_df
	'''

	param = {'max_depth':10, 'eta':0.1, 'min_child_weight':1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'logloss', 'subsample':1, 'col_sample_bytree':0.8 }

	#specify validation set to watch performance
	watchlist  = [(train_df,'train'), (eval_DMatrix,'eval')]
	num_round = 2000
	
	bst = xgb.train(param, train_DMatrix, num_round, watchlist, early_stopping_rounds=100)

	#prediction
	test_DMatrix = xgb.DMatrix(test_df)
	preds = bst.predict(test_DMatrix, ntree_limit=bst.best_ntree_limit)
	write_results(preds, ids_test) #need to get ids from somewhere






