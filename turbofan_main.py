import pandas as pd
from tseries import TimeSeriesFeaturizer
import numpy as np
import glob
import tqdm


def get_dfs_by_machine(file):
	df_file = pd.read_csv(file)
	gb = df_file.groupby('machine')
	dfs = [group.iloc[:, 2:] for _, group in gb]
	return dfs


dfs = list()
dfs_test = list()
data_dir = '/home/joanes/GBL/data/Turbofan/'
files = glob.glob(data_dir + 'train*.csv')
files_test = glob.glob(data_dir + 'test*.csv')
for file, file_test in zip(files, files_test):
	dfs.extend(get_dfs_by_machine(file))
	dfs_test.extend(get_dfs_by_machine(file_test))

tseries = TimeSeriesFeaturizer()

transformed = tseries.featurize(dfs)

trans_test = tseries.test_featurization(dfs_test)
