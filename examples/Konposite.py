# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import itertools
import pickle

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


PATH = '/media/joanes/0BB3-1FA1/CSV_DATA/'
files = glob(f'{PATH}*.csv')


# In[3]:


# Helper functions
def rf_feat_importance(m, df):
	return pd.DataFrame({'cols': df.columns, 'imp': m.feature_importances_}).sort_values('imp', ascending=False)


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()


def evaluate(model, test_features, test_labels):
	accuracy = m.score(test_features, test_labels)
	mse = mean_squared_error(test_labels, model.predict(test_features))

	print('Model Performance')
	print('Accuracy = {:0.4f}%.'.format(accuracy))
	print('Mean Squared Error = {:0.4f}%.'.format(mse))
	return accuracy


# In[4]:


def get_df_list_and_target(files):
	# Get only those values that are available before the analysis
	filter_cols = ['Total_UnfilledZones', 'Total_FillingQuality', 'TOTAL_QUALITY', 'Total_PorosityQuantity',
				   'Total_PorosityQuality', 'Time']
	df_list = list()
	target = list()
	for file in files:
		df = pd.read_csv(file)
		target.append(df.TOTAL_QUALITY.unique()[0])
		df_filtered = df.drop(axis=1, columns=filter_cols)
		filter_col = [col for col in df_filtered if
					  not col.endswith(('VoidContent', 'VoidQuality', 'Filling', 'FillingQuality'))]
		df_filtered = df_filtered[filter_col]
		df_list.append(df_filtered)

	target = pd.DataFrame(target, columns=['valid'])
	return df_list, target


# In[ ]:


model_ratio = 0.02
model_size = int(len(files) * model_ratio)

#df_list, target_model = get_df_list_and_target(files[:model_size])

print('Modelorako fitxategiak hartuta')

# In[5]:


# Add library's path to notebook
import os
import sys

sys.path.append('../../time-series-featurizer/')

print('Tseries loaded')

# In[ ]:
#
#
# from ts_featurizer import TimeSeriesFeaturizer
#
# tseries = TimeSeriesFeaturizer()
# model = tseries.featurize(df_list, n_jobs=8)
#
# print('Modeloa sortuta')

# In[ ]:
#
#
# import pickle
#
# filehandler = open('tmp/tseries.pickle', 'wb')
# pickle.dump(tseries, filehandler)

# In[6]:


with open('tmp/tseries.pickle', 'rb') as filehandler:
	tseries = pickle.load(filehandler)

print('Tseries model loaded')

# In[ ]:


# tseries._featurized_data


# In[ ]:


import os

os.makedirs('tmp', exist_ok=True)
train_ratio = 0.1
train_size = int(len(files) * train_ratio)
for time in range(8, 11):
	df_list, target_featurized = get_df_list_and_target(files[(time - 1) * train_size: time * train_size])
	print(f'Loaded DataFrame lists len is {len(df_list)}, from {(time - 1) * train_size} to {time * train_size}')

	featurized = tseries.featurize(df_list, n_jobs=8, apply_model=True)
	featurized.reset_index(drop=True).to_feather(f'tmp/featurized_{time}')
	target_featurized.reset_index(drop=True).to_feather(f'tmp/target_featurized_{time}')
	print('Stored the featurized files')

# In[ ]:


features = pd.concat([model, featurized])  # It's important to do it in the original order
target = pd.concat([target_model, target_featurized])
print(features.dtypes.unique())
print(f'NaN features: {features.isna().sum().sum()}')
na_cols = features.columns[features.isna().any()].tolist()
features.drop(axis=1, columns=na_cols, inplace=True)
print(f'NaN features: {features.isna().sum().sum()}')


# In[ ]:


def split_vals(df, n): return df[:n], df[n:]


X_train, X_test = split_vals(features, train_size)
y_train, y_test = split_vals(target, train_size)

# ## Store featurized dataframe to feather files

# In[ ]:


import os

os.makedirs('tmp', exist_ok=True)
X_test.reset_index(drop=True).to_feather('tmp/X_test')
X_train.reset_index(drop=True).to_feather('tmp/X_train')
y_train.reset_index(drop=True).to_feather('tmp/y_train')
y_test.reset_index(drop=True).to_feather('tmp/y_test')

# ## Read feather file into dataframe

# In[ ]:


# import feather

# X_test = feather.read_dataframe('tmp/X_test')
# X_train = feather.read_dataframe('tmp/X_train')
# y_train = feather.read_dataframe('tmp/y_train')
# y_test = feather.read_dataframe('tmp/y_test')


# In[ ]:


# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# m = RandomForestClassifier(n_estimators=80,max_features=0.5, min_samples_leaf=10, oob_score=True, n_jobs=-1)
# m.fit(X_train, y_train.values.ravel())
# print(f'Training score: {m.score(X_train, y_train.values.ravel())}')
# print(f'Testing score: {m.score(X_test, y_test.values.ravel())}')
# print(f'Out of bag score: {m.oob_score_}')
# print('\n')
# evaluate(m, X_test, y_test);


# In[ ]:


# y_pred = m.predict(X_test)

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_pred, y_test)

# # Plot non-normalized confusion matrix

# plot_confusion_matrix(cnf_matrix, classes=[ 'Desegokia', 'Egokia'], title='Normalizatu gabeko konfusio matrizea')

# plt.show()


# ## Get initial classifier's score

# In[ ]:


# m = RandomForestClassifier(n_estimators=80,max_features=0.5, min_samples_leaf=10, oob_score=True, n_jobs=-1)
# m.fit(train, target[:train_size].values.ravel())
# print(f'Training score: {m.score(train, target[:train_size].values.ravel())}')
# print(f'Testing score: {m.score(test, target[train_size:].values.ravel())}')
# print(f'Out of bag score: {m.oob_score_}')
# print('\n')
# evaluate(m, test, target[train_size:]);


# ## Get classifier's confusion matrix

# In[ ]:


# y_test = m.predict(test)

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, target[train_size:])

# # Plot non-normalized confusion matrix

# plot_confusion_matrix(cnf_matrix, classes=[ 'Desegokia', 'Egokia'], title='Normalizatu gabeko konfusio matrizea')

# plt.show()


# ## Feature importance

# In[ ]:


# df_raw = pd.concat([train, test])
# df_raw.shape


# In[ ]:


# fi = rf_feat_importance(df=train, m=m)

# fi[:50].plot('cols', 'imp', 'barh', figsize=(20,7))
# plt.show()

# df_important = df_raw[fi[:50].cols]


# ## Check redundancy between features

# In[ ]:


# from scipy.cluster import hierarchy as hc

# corr = np.round(scipy.stats.spearmanr(df_important).correlation, 4)
# corr_condensed = hc.distance.squareform(1-corr)
# z = hc.linkage(corr_condensed, method='average')
# fig = plt.figure(figsize=(16,10))
# dendrogram = hc.dendrogram(z, labels=df_important.columns, orientation='left', leaf_font_size=16)
# plt.show()


# ### Here we can see that many important features are very highly correlated, so we will try to run the classifier without those who are redundant (only deleting one of both redundant cols)

# In[ ]:


# redundants = ['Zone12_Pressure_Time_numpy.median', 'Zone16_Pressure_Time_scipy.stats.kurtosis', 'Zone24_Pressure_Time_scipy.stats.kurtosis',
#             'Zone24_tfilling_Time_numpy.median','Zone24_tfilling_Time_numpy.min', 'Zone24_tfilling_Time_numpy.mean',
#              'Zone24_tfilling_Time_numpy.max', 'Zone24_tfilling_Time_numpy.percentile_95']


# for red in redundants:
#     m = RandomForestClassifier(n_estimators=80,max_features=0.5, min_samples_leaf=10, oob_score=True, n_jobs=-1)
#     wo_red = df_important.drop(axis=1, columns=[red])
#     m.fit(wo_red[:train_size], target[:train_size].values.ravel())
#     print(f'{red}:\n')
#     print(f'\tTraining score: {m.score(wo_red[:train_size], target[:train_size].values.ravel())}')
#     print(f'\tTesting score: {m.score(wo_red[train_size:], target[train_size:].values.ravel())}')
#     print(f'\tOut of bag score: {m.oob_score_}\n')

# m = RandomForestClassifier(n_estimators=80,max_features=0.5, min_samples_leaf=10, oob_score=True, n_jobs=-1)
# df_wo_redundant = df_important.drop(axis=1, columns=redundants)
# m.fit(df_wo_redundant[:train_size], target[:train_size].values.ravel())

# print(f'Without redundants:\n')
# print(f'\tTraining score: {m.score(df_wo_redundant[:train_size], target[:train_size].values.ravel())}')
# print(f'\tTesting score: {m.score(df_wo_redundant[train_size:], target[train_size:].values.ravel())}')
# print(f'\tOut of bag score: {m.oob_score_}\n')

# base_accuracy = evaluate(m, df_wo_redundant[train_size:], target[train_size:].values.ravel())


# ### We can see that without those redundant features, our model is still quite solid, so we remove them

# ## GridSearch in order to calculate best hiperparameters

# In[ ]:


# def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
#     # Get Test Scores Mean and std for each grid search
#     scores_mean = cv_results['mean_test_score']
#     scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

#     scores_sd = cv_results['std_test_score']
#     scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

#     # Plot Grid search scores
#     figure, ax = plt.subplots(1,1, figsize=(15,6))

#     # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
#     for idx, val in enumerate(grid_param_2):
#         ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

#     ax.set_title("Grid Search Scores", fontsize=18, fontweight='bold')
#     ax.set_xlabel(name_param_1, fontsize=16)
#     ax.set_ylabel('CV Average Score', fontsize=16)
#     ax.legend(loc="best", fontsize=10)
#     ax.grid(True)


# In[ ]:


# n_estimators = [80, 100, 160, 200]
# max_features = [0.5, 'log2', 'sqrt', 'auto']
# grid_search = GridSearchCV(m,
#             dict(n_estimators=n_estimators,
#                  max_features=max_features),
#                  cv=5, n_jobs=8)

# grid_search.fit(features[:train_size], target[:train_size].values.ravel());
# grid_search.best_params_


# In[ ]:


# plot_grid_search(grid_search.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')


# In[ ]:


# best_grid = grid_search.best_estimator_
# grid_accuracy = evaluate(best_grid, features[train_size:], target[train_size:])


# ### So, we see how the score of the max_features=0.5 is the best, and is quite constant when it comes to n_estimators. Now, we choose max_features=0.5, and we try to 

# In[ ]:


# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [0.5],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300]
# }
# clf = RandomForestClassifier()

# pipe_grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=4, n_jobs=-1, verbose=2)

# pipe_grid.fit(features[:train_size], target[:train_size].values.ravel());


# In[ ]:


# best_grid = pipe_grid.best_estimator_
# grid_accuracy = evaluate(best_grid, features[train_size:], target[train_size:])


# ## We check the how the feature importance is now and the redundat features

# In[ ]:


# fi = rf_feat_importance(df=features[:train_size], m=best_grid)

# fi[:60].plot('cols', 'imp', 'barh', figsize=(20,7))
# plt.show()


# In[ ]:


# from scipy.cluster import hierarchy as hc

# corr = np.round(scipy.stats.spearmanr(df_wo_redundant).correlation, 4)
# corr_condensed = hc.distance.squareform(1-corr)
# z = hc.linkage(corr_condensed, method='average')
# fig = plt.figure(figsize=(16,10))
# dendrogram = hc.dendrogram(z, labels=df_wo_redundant.columns, orientation='left', leaf_font_size=16)
# plt.show()
