import pandas as pd
import numpy as np
from config import default_conf
from tools import get_attr_from_module, parallel_process, config_dict_validation
import copy


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
		self._config = copy.deepcopy(config)
		self._featurized_data = pd.DataFrame()
		self._testing_data = pd.DataFrame()
		self._prev_trans = dict()

	@staticmethod
	def _check_dataset(data, time_column, ):
		"""
		Dataset validation is performed in this function, checking NaNs, data type, index's validity...


		:param data: dataFrame to be validated.
		:param time_column: Date column of the DataFrame.
		:return: validated data, with some changes in index, etc.
		"""
		if not isinstance(data, pd.DataFrame):
			raise TypeError('Data must be a pandas DataFrame.')

		if data.isna().sum().sum() > 0:
			raise Exception('NaNs have been found in the data.')
		if not time_column:
			if not data.index.is_all_dates:
				try:
					date_typed = pd.to_datetime(data.index.values, unit='s')
					data.set_index(date_typed, inplace=True)

				# TODO Honi verbosity nibel bat jarri bihar jako
				#  print('Index was used as the time reference column.')

				except Exception as e:
					raise Exception('No time column was specified, and DataFrame\'s index values are not dates.')

		else:

			if time_column not in data.select_dtypes(include=[np.datetime64, ]).columns:
				try:
					data[time_column] = pd.to_datetime(data[time_column], unit='s')
					print('Time column was converted from numeric to date type.')

				except Exception as e:
					raise Exception('Specified time column is not of datetime type.')

			data.set_index(time_column, inplace=True)

		data = data.select_dtypes(include=[np.number])
		data.sort_index(inplace=True)
		data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)

		return data

	@staticmethod
	def _featurize_each_class(data, config, use_dask, n_jobs):
		"""

		Creates an instance of the class specified in the configuration, and calls its featurize function.
		This class must inherit the BaseFeaturizer class.

		For example:

		featurizer = TimeFeaturizer(data, config)
		featurizer.featurize()


		:param data: The data to featurize.
		:param config: The configuration for the featurization, with all the functions to apply.
		:param use_dask: Specifies if the library for parallel computing is used. (Not implemented)
		:param n_jobs: Specifies the number of cores used for the parallelization.
		:return:

		"""
		featurizer = config.class_(data, config)
		transf, prev_transf = featurizer.featurize(use_dask, n_jobs)

		return transf, (config.class_, prev_transf)

	def featurize(self, train, time_column=None, use_dask=False, n_jobs=1):
		"""
		Main function, comparable to fit in sklearn, data and configuration is validated, and with each class
		specified in the configuration, featurization is completed.

		Firstly, prepares the configuration, separating each featurizer in a dictionary with the specified settings.

		Each featurizer contains the previous transformations to apply to the data, the features to extract
		(with function name, args, kwargs, etc.) and some more settings.

		After the configuration is set, the parallel_process function is called, where _featurize_each_class function
		will be parallelly called with the specified configuration and n_jobs.

		This returns a list of tuples, with the previous transformation and the featurized data,
		and finally these are stored in their own class attributes.

		:param train: dataFrame used for the training. Featurization is applied to this data.
		:param time_column: The date column in order the time series to be ordered properly.
		:param use_dask: Specifies if the library for parallel computing is used. (Not implemented)
		:param n_jobs: Specifies the number of cores used for the parallelization.
		:return: featurized DataFrame.
		"""
		data = self._check_dataset(train, time_column, )
		self._config, errors = config_dict_validation(self._config)

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

	def test_featurization(self, test, time_column=None, use_dask=False, n_jobs=1):
		"""
		This function takes the previously (in featurize function) featurized dataset's model, and applies to a new dataset.

		:param test: dataFrame used for the testing. Featurization and the model is applied to this data.
		:param time_column: The date column in order the time series to be ordered properly.
		:param use_dask: Specifies if the library for parallel computing is used. (Not implemented)
		:param n_jobs: Specifies the number of cores used for the parallelization.
		:return:
		"""

		test = self._check_dataset(test, time_column, )
		transf_list = list()
		prev_trans_list = dict()
		f_data = self._featurized_data

		for featurizer in self._config:
			test_executions = featurizer.get_test_features()
			for execution in test_executions:

				ret = f_data[featurizer.name].loc[:, f_data[featurizer.name].columns.str.startswith(str(execution))]

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

	def get_previous_trans(self):
		"""
		Getter function for the transformations completed before the featurizations.

		:return: The transformed data created for featurization on each featurizer.
		"""
		return self._prev_trans

	def get_featurized_data(self):
		"""
		Getter function for featurized data.
		:return:
		"""
		return self._featurized_data
