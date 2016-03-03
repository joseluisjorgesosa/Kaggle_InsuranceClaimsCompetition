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

	# #replace missing values with -999
	df.fillna(-999, inplace=True)

	#encode categorical features as integers
	cat_columns = df.select_dtypes(['object']).columns
	for c in cat_columns:
		df[c] = df[c].astype('category')

	df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes) #this may not be strictly necessary, since we already convert to category type first

	return id_col, df



if __name__=="__main__":

	#load data
	ids_train, df = load_data(sys.argv[1])
	ids_test, predict_df = load_data(sys.argv[2]) #data set to predict values for

	traindf, testdf = train_test_split(df, test_size = 0.2)

	features = list(traindf.columns[1:])
	labels = traindf.columns[0]

	train_DMatrix = xgb.DMatrix(traindf[features], traindf[labels])
	test_DMatrix = xgb.DMatrix(testdf[features], testdf[labels]) #for validation/early stopping

	predict_DMatrix = xgb.DMatrix(predict_df)

	#specify parameters
	param = {'max_depth':10, 'eta':0.1, 'silent':1, 'objective':'reg:linear' }

	#specify validations set to watch performance
	watchlist  = [(test_DMatrix,'eval'), (train_DMatrix,'train')]
	num_round = 70
	bst = xgb.train(param, train_DMatrix, num_round, watchlist)

	#prediction
	preds = bst.predict(predict_DMatrix)

	test_submission = open('results.csv','w')
	test_submission.write('ID,PredictedProb\n')

	for i in range(len(preds)):
		test_submission.write(str(ids_test[i]) + "," + str(preds[i])+"\n")

	test_submission.close()

	# print preds
	# print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
	bst.save_model('0001.model')