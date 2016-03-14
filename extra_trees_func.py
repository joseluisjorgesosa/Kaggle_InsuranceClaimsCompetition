#!/usr/bin/python

import sys
import math
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier)
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split




def load_data_rf(data_file):
	'''
	Load the training and testing data
	'''
	df = pd.read_csv(data_file)

	#separate id column
	id_col = df['ID']
	cols = df.columns.tolist()[1:]
	df = df[cols]

	# #replace missing values with -999
	#df.fillna(-999, inplace=True)
	df.fillna(-999, inplace=True)

	#encode categorical features as integers
	cat_columns = df.select_dtypes(['object']).columns
	for c in cat_columns:
		df[c] = df[c].astype('category')

	df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes) #this may not be strictly necessary, since we already convert to category type first

	return id_col, df


def get_predictions(train_file, test_file):
	print "about to start"

	#load data
	ids_train, df = load_data_rf(train_file)
	ids_test, predict_df = load_data_rf(test_file) #data set to predict values for

	traindf, testdf = train_test_split(df, random_state=1, test_size = 0.2)

	features = list(traindf.columns[1:])
	labels = traindf.columns[0]

	X = traindf[features]
	Y = traindf[labels]
	X_test = testdf[features]
	Y_test = testdf[labels]

	print "about to create classifier"

	#clf = linear_model.LogisticRegression(C=1e5)
	n_features = len(features)
	sqrt_features = round(math.sqrt(n_features))
	print n_features

	clf = Pipeline([
	  ('feature_selection', SelectFromModel(
	  	ExtraTreesClassifier(verbose=1, n_jobs=-1, n_estimators=700,max_features= 'sqrt',criterion= 'entropy',min_samples_split= 1,
	                            max_depth= 50, min_samples_leaf= 1), threshold='0.5*mean')),
	  ('classification', ExtraTreesClassifier(verbose=1, n_jobs=-1, n_estimators=700,max_features= 'sqrt',criterion= 'entropy',min_samples_split= 1,
	                            max_depth= 50, min_samples_leaf= 1) 
		)
	])
	print "about to fit"
	clf.fit(X, Y)

	features = clf.named_steps['feature_selection'].get_support(indices=True)
	print features
	print len(features)

	#scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
	#cross_score = scores.mean()
	cross_score = 0
	print "about to predict"
	y_predict_train = clf.predict(X)
	y_predict_test = clf.predict(X_test)

	print "done predicting"
	score = clf.score(X_test,Y_test)
	mse_train = mean_squared_error(y_predict_train, Y)
	mse_test = mean_squared_error(y_predict_test, Y_test)
	print("train: " + str(mse_train) + " cross_score : " +  str(cross_score) + " test: " + str(mse_test) + " score: " + str(score))
	

	#prediction for testing data
	preds = clf.predict_proba(predict_df)

	predictions = [y_predict_train, y_predict_test, preds]

	return predictions

'''	

if __name__=="__main__":

	print "about to start"
	#load data
	ids_train, df = load_data(sys.argv[1])
	ids_test, predict_df = load_data(sys.argv[2]) #data set to predict values for

	traindf, testdf = train_test_split(df, test_size = 0.2)

	features = list(traindf.columns[1:])
	labels = traindf.columns[0]

	X = traindf[features]
	Y = traindf[labels]
	X_test = testdf[features]
	Y_test = testdf[labels]

	#clf = AdaBoostClassifier(n_estimators=100)
	print "about to create classifier"
	#clf = svm.SVC(probability=True)
	#clf = SGDClassifier(loss="log", penalty="l2")

	#clf = linear_model.LogisticRegression(C=1e5)
	n_features = len(features)
	sqrt_features = round(math.sqrt(n_features))
	print n_features
	#clf =  RandomForestClassifier(n_estimators=300, max_features='sqrt',max_depth=None, min_samples_split=1)
	#clf =  ExtraTreesClassifier(n_jobs=-1, n_estimators=700, max_features='sqrt',max_depth=None, min_samples_split=1)
	#clf =  ExtraTreesClassifier(verbose=1, n_jobs=-1, n_estimators=700,max_features= 50,criterion= 'entropy',min_samples_split= 5,
	 #                           max_depth= 50, min_samples_leaf= 5) 

	#clf =  ExtraTreesClassifier(verbose=1, n_jobs=-1,criterion= 'entropy',
     #                        n_estimators=200, max_features=50, min_samples_split= 10, 
      #                       max_depth=50, min_samples_leaf= 10) 

	clf = Pipeline([
	  ('feature_selection', SelectFromModel(
	  	ExtraTreesClassifier(verbose=1, n_jobs=-1, n_estimators=700,max_features= 'sqrt',criterion= 'entropy',min_samples_split= 1,
	                            max_depth= 50, min_samples_leaf= 1), threshold='0.5*mean')),
	  ('classification', ExtraTreesClassifier(verbose=1, n_jobs=-1, n_estimators=700,max_features= 'sqrt',criterion= 'entropy',min_samples_split= 1,
	                            max_depth= 50, min_samples_leaf= 1) 
		)
	])
	print "about to fit"
	clf.fit(X, Y)

	features = clf.named_steps['feature_selection'].get_support(indices=True)
	print features
	print len(features)
	#scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
	#cross_score = scores.mean()
	cross_score = 0
	print "about to predict"
	y_predict_train = clf.predict(X)
	y_predict_test = clf.predict(X_test)

	print "done predicting"
	score = clf.score(X_test,Y_test)
	mse_train = mean_squared_error(y_predict_train, Y)
	mse_test = mean_squared_error(y_predict_test, Y_test)
	print("train: " + str(mse_train) + " cross_score : " +  str(cross_score) + " test: " + str(mse_test) + " score: " + str(score))
	#clf.predict(predict_df)


	#prediction for testing data
	preds = clf.predict_proba(predict_df)

	test_submission = open('results.csv','w')
	test_submission.write('ID,PredictedProb\n')

	for i in range(len(preds)):
		pr = preds[i]
		#print pr[1]
		test_submission.write(str(ids_test[i]) + "," + str(pr[1])+"\n")

	test_submission.close()

	# print preds
	# print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
	#bst.save_model('0001.model')

'''

