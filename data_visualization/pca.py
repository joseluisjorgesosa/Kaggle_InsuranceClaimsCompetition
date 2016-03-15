from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

#Function: load_data
def load_data(data_file):
	'''
	Load the training and testing data
	'''
	df = pd.read_csv(data_file)

	#separate id column
	id_col = df['ID']
	cols = df.columns.tolist()[1:]
	df = df[cols]

	#replace missing values with -999
	df.fillna(0, inplace=True)

	#encode categorical features as integers
	cat_columns = df.select_dtypes(['object']).columns
	for c in cat_columns:
		df[c] = df[c].astype('category')

	df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes) #this may not be strictly necessary, since we already convert to category type first

	#Replace missing value with average value (by column)
	#col_labels = df.columns[0:]
	#for col_label in col_labels:
	#	df[col_label].fillna(df[col_label].mean(),inplace=True)
	

	return id_col, df

#Function: getTrainingData()
#
#Description: Loads the data and then puts it into numpy arrays
#
# Returns training and validation data to user
def get_all_data(training_file): 
	ids_train, df = load_data(training_file)


	features = list(df.columns[1:])
	labels = df.columns[0]

		
	#Read in X training data
	X= np.zeros((len(df),len(features)))
	X[0:len(df),0:len(features)] = (np.asarray(df.iloc[0:len(df),1:])-np.mean(np.asarray(df.iloc[0:len(df),1:]),axis=0))/np.std(np.asarray(df.iloc[0:len(df),1:]),axis=0)

	Y = np.array(df[labels])

	
	return X,Y


if __name__ == "__main__":


	X,Y = get_all_data(sys.argv[1])	
	fig = plt.figure(1, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	X_reduced = PCA(n_components=3).fit_transform(X)

	ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,marker='o')

	plt.show()

