import pandas as pd
from tseries import TimeSeriesFeaturizer
import numpy as np
import glob
import tqdm

data_dir = '/home/joanes/GBL/data/Turbofan/'
threshold = .0000001
chunk_size = 30
files = glob.glob(data_dir + 'train*.csv')
machine_ids = list()

file_trans_df_list = list()
for file in files:
	total_trans_list = list()
	machine_ids_file = list()

	df_file = pd.read_csv(file)
	gb = df_file.groupby('machine')
	dfs = [group for _, group in gb]
	for df in tqdm.tqdm(dfs[:3]):
		trans_list = list()
		machine_ids_file.append(df['machine'].iloc[0])
		df_filtered = df

		for column in df_filtered.iloc[:, 2:]:
			df_col = df_filtered[[column]]
			splitted_col = np.array_split(df_col, df_filtered.shape[0] / chunk_size)
			splitted_col = [chunk.reset_index(drop=True) for chunk in splitted_col]
			splitted_df = pd.concat(splitted_col, axis=1, ignore_index=True)
			splitted_df.dropna(how='any', inplace=True)

			tseries = TimeSeriesFeaturizer()

			transformed = tseries.featurize(splitted_df)
			trans_list.append(transformed)

		total_trans_list.append(pd.concat(trans_list, keys=df_filtered.iloc[:, 2:].columns, axis=1))
	file_trans_df_list.append(pd.concat(total_trans_list, keys=set(machine_ids_file), axis=1))
	machine_ids.extend(machine_ids_file)

last_df = pd.concat(file_trans_df_list, keys=set(machine_ids))
print(last_df)
