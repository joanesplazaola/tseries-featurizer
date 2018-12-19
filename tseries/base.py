import pandas as pd
import numpy as np
from config import default_conf
from tools import get_attr_from_module, parallel_process


class TimeSeriesFeaturizer:

	def __init__(self, config=default_conf, ):
		self._config = config
		self._t_data = pd.DataFrame()
		self._prev_trans = dict()  # Here previous transformations will be stored with its key

	# TODO NaNekin ze egin.. Dena ondo dagoela begiratu
	# check if data is dataframe or series
	def _check_dataset(self, data, time_column, ):

		# TODO Hemen begiratu behar da ea mantentzen duen konfigurazio fitxategiaren egitura minimoa
		if not isinstance(self._config, dict):
			raise Exception('Configuration file must be a dictionary.')
		if data.isna().sum().sum() > 0:
			raise Exception('NaNs have been found in the data.')
		if not time_column:
			if not data.index.is_all_dates:
				raise Exception('No time column was specified, and DataFrame\'s index values are not dates.')
			print('Index was used as the time reference column.')
		else:
			if time_column not in data.select_dtypes(include=[np.datetime64]).columns:
				raise Exception('Specified time column is not of datetime type.')

			data.set_index(time_column, inplace=True)

		data = data.select_dtypes(include=[np.number])
		data.sort_index(inplace=True)
		data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)

		return data

	def featurize(self, data, time_column=None, use_dask=False, n_jobs=1):

		data = self._check_dataset(data, time_column, )

		kwargs = [
			{'key': key, 'data': data, 'use_dask': use_dask, 'n_jobs': n_jobs}
			for key in self._config['train'].keys()
		]

		ret = parallel_process(kwargs, self._featurize_each_class, use_kwargs=True, n_jobs=n_jobs)

		transf_list, prev_trans = map(list, zip(*ret))
		self._t_data = pd.concat(transf_list, axis=1, keys=list(self._config['train'].keys()))

		self._prev_trans = [{'name': trans[0], 'value': trans[1]} for trans in prev_trans]

		return self._t_data

	def _featurize_each_class(self, key, data, use_dask, n_jobs):
		my_class = get_attr_from_module(self._config['train'][key]['class'])
		featurizer = my_class(data, self._config['train'][key])
		transf, prev_transf = featurizer.featurize(use_dask, n_jobs)

		return transf, (key, prev_transf)


def get_previous_trans(self):
	return self._prev_trans


def get_featurized_data(self):
	return self._t_data
