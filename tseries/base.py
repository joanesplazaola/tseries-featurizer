import pandas as pd
import numpy as np
from config import default_conf
from tools import get_attr_from_module, parallel_process, config_dict_validation


class TimeSeriesFeaturizer:
	"""
	Class with the Time Series data (pandas Dataframe) and the featurization configuration file's content.

	...

	Attributes
	----------
	_config : dict
		Configuration dictionary with all the methods, args and kwargs to apply to the featurization.

	_featurized_data : pd.DataFrame
		Featurized data.

	_prev_trans : dict
		Data's state in each one of the tranformations done previous to the featurization.

	Methods
	-------
	_check_dataset(data, time_column)
		The validity of the dataset is checked.
	featurize(train, time_column=None, use_dask=False, n_jobs=1)
		Takes training data, and featurizes it applying the functions and transformations specified in the config file.
	_featurize_each_class(self, key, data, use_dask, n_jobs)
		Takes training data, and the transformation name, and applies the specified featurization.
	get_previous_trans()
		Getter function of the transformed data used for the featurization(self._prev_trans)
	get_featurized_data()
		Getter function of the featurized data(self._t_data)

	"""

	def __init__(self, config=default_conf, ):
		self._config = config
		self._featurized_data = pd.DataFrame()
		self._testing_data = pd.DataFrame()

		self._prev_trans = dict()  # Here previous transformations will be stored with its key

	# TODO NaNekin ze egin.. Dena ondo dagoela begiratu
	# check if data is dataframe or series
	def _check_dataset(self, data, time_column, ):
		"""

		:param data:
		:param time_column:
		:return:
		"""

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

	def featurize(self, train, time_column=None, use_dask=False, n_jobs=1):
		"""

		:param train:
		:param time_column:
		:param use_dask:
		:param n_jobs:
		:return:
		"""
		data = self._check_dataset(train, time_column, )
		errors, self._config = config_dict_validation(self._config)

		if len(errors) > 0:
			self.config_errors = errors
			raise Exception(
				'Error(s) in configuration dictionary, in order to correct them, check config_errors variable'
			)

		kwargs = [
			{'config': featurizer, 'data': data, 'use_dask': use_dask, 'n_jobs': n_jobs}
			for featurizer in self._config
		]

		ret = parallel_process(kwargs, self._featurize_each_class, use_kwargs=True, n_jobs=n_jobs)

		transf_list, prev_trans = map(list, zip(*ret))
		keys = [featurizer.name for featurizer in self._config]
		self._featurized_data = pd.concat(transf_list, axis=1, keys=keys)

		self._prev_trans = [{'name': trans[0], 'value': trans[1]} for trans in prev_trans]

		return self._featurized_data

	def test_features(self, test, time_column=None, use_dask=False, n_jobs=1):
		"""

		:param test:
		:param time_column:
		:param use_dask:
		:param n_jobs:
		:return:
		"""

		test = self._check_dataset(test, time_column, )
		transf_list = list()
		prev_trans_list = dict()

		for featurizer in self._config:
			test_executions = featurizer.get_test_features()
			for execution in test_executions:
				ret = self._featurized_data[featurizer.name][str(execution)]
				for preprocessor, kwargs in execution.test['preprocessors'].items():
					func = get_attr_from_module(preprocessor)
					ret = func(ret, **kwargs)
				execution.kwargs['fit_model'] = ret
				execution.kwargs['test'] = True

			transf, prev_trans = self._featurize_each_class(test, featurizer, use_dask, n_jobs)
			transf_list.append(transf)
			prev_trans_list[prev_trans[0]] = prev_trans[1]

		keys = [featurizer.name for featurizer in self._config]
		self._testing_data = pd.concat(transf_list, axis=1, keys=keys)
		return self._testing_data

	def _featurize_each_class(self, data, config, use_dask, n_jobs):
		"""

		:param key:
		:param data:
		:param use_dask:
		:param n_jobs:
		:return:
		"""
		featurizer = config.class_(data, config)
		transf, prev_transf = featurizer.featurize(use_dask, n_jobs)

		return transf, (config.class_, prev_transf)

	def get_previous_trans(self):
		"""

		:return:
		"""
		return self._prev_trans

	def get_featurized_data(self):
		"""

		:return:
		"""
		return self._featurized_data
