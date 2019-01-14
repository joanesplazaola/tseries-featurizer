import pandas as pd
import numpy as np
from config import default_conf
from tools import get_attr_from_module, parallel_process, config_dict_validation
import copy
import tqdm


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

	_testing_data : pd.DataFrame
		Featurized test data.

	_prev_trans : dict
		Training data's state in each one of the tranformations done previous to the featurization.

	_prev_trans_test : dict
		Testing data's state in each one of the tranformations done previous to the featurization.

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
		self._prev_trans_test = dict()

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
					date_typed = pd.to_datetime(np.array(range(len(data.index.values))), unit='s')
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

		:param data: The data to featurize.
		:param config: The configuration for the featurization, with all the functions to apply.
		:param use_dask: Specifies if the library for parallel computing is used. (Not implemented)
		:param n_jobs: Specifies the number of cores used for the parallelization.
		:return:



		"""

		featurizer = config.class_(config)
		transf, prev_transf = featurizer.featurize(data, use_dask, n_jobs)

		return transf, (config.class_, prev_transf)

	@staticmethod
	def check_all_same_columns(df_list):
		"""
		Checks if in a list of pd.DataFrame all DataFrame s have the same columns.
		:param df_list: list of pd.Dataframes to check.
		:return: Whether all the DataFrames have same columns.
		"""
		cols = df_list[0].columns.tolist()

		return all(x.columns.tolist() == cols for x in df_list)

	def check_datasets(self, df, time_column):
		"""
		Validates and formats list of dataframes.

		:param df: list of DataFrames or DataFrame
		:param time_column:
		:return: formatted list of DataFrames
		"""
		dfs = list()

		if isinstance(df, pd.DataFrame):
			df = [df]
		if not self.check_all_same_columns(df):
			raise Exception('Dataframes in list are not comparable as have different columns.')

		for df in df:
			dfs.append(self._check_dataset(df, time_column, ))

		return dfs

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

		config, errors = config_dict_validation(self._config)

		if len(errors) > 0:
			self.config_errors = errors
			raise Exception(
				'Error(s) in configuration dictionary, in order to correct them, check config_errors attribute'
			)
		dfs = self.check_datasets(train, time_column)
		huge_df = pd.concat(dfs, keys=range(len(dfs)))
		transf_column_list = list()
		for column in tqdm.tqdm(huge_df.columns):
			grouped = huge_df[[column]].groupby(level=0)
			column_df = self.get_column_df_from_groups(grouped)

			kwargs = [
				{'config': featurizer, 'data': column_df, 'use_dask': use_dask, 'n_jobs': n_jobs}
				for featurizer in config
			]

			ret = parallel_process(kwargs, self._featurize_each_class, use_kwargs=True, n_jobs=n_jobs)

			transf_list, prev_trans = map(list, zip(*ret))

			transf_column_list.append(pd.concat(transf_list, axis=1, keys=[f.name for f in config]))

			self._prev_trans[column] = {trans[0].__name__: trans[1] for trans in prev_trans}
		self._featurized_data = pd.concat(transf_column_list, keys=huge_df.columns.values, axis=1)
		return self._featurized_data

	def apply_functions_to_data(self, preprocessors, ret):
		"""
		Recursive function that applies a list of functions to a DataFrame
		:param preprocessors:
		:param ret:
		:return:
		"""
		prep = copy.deepcopy(preprocessors)
		if not prep:
			return ret
		fn = list(prep.keys())[0]
		kwargs = prep.pop(fn)
		func = get_attr_from_module(fn)
		if not prep:
			return func(ret, **kwargs)
		return self.apply_functions_to_data(prep, func(ret, **kwargs))

	@staticmethod
	def get_column_df_from_groups(grouped):
		"""
		Creates a DataFrame from grouped DataFrames.

		This is used in order to group the variables of the DataFrames, so we will get a DataFrame from each
		variable of the DataFrame.

		For example, if we have two sensors that measure the same things, for each variable they measure, a Dataframe
		will be generated, grouping both sensor's  variables.

		:param grouped: Group of Series (these are the same variable of different machine/sensors)
		:return: DataFrame created with those Series.



		"""
		column_df = pd.DataFrame()

		for group_name, group in grouped:
			group.index = group.index.droplevel(0)
			group.columns = [group_name]
			column_df = pd.concat([column_df, group], axis=1, )
		return column_df

	def set_test_arguments(self, featurizer, f_data, column):
		test_executions = featurizer.get_test_features()
		for execution in test_executions:
			ret = f_data[featurizer.name].loc[:, f_data[featurizer.name].columns.str.startswith(str(execution))]
			ret = self.apply_functions_to_data(execution.test['preprocessors'], ret)

			execution.kwargs['fit_model'] = ret
			execution.kwargs['test'] = True
		if featurizer.test:
			featurizer.train_prev_data = self._prev_trans[column][featurizer.class_.__name__]
		return featurizer

	def test_featurization(self, test, time_column=None, use_dask=False, n_jobs=1):
		"""
		This function takes the previously (in featurize function) featurized dataset's model, and applies to a new dataset.

		:param test: dataFrame used for the testing. Featurization and the model is applied to this data.
		:param time_column: The date column in order the time series to be ordered properly.
		:param use_dask: Specifies if the library for parallel computing is used. (Not implemented)
		:param n_jobs: Specifies the number of cores used for the parallelization.
		:return:
		"""

		config, errors = config_dict_validation(self._config, test=True)

		if len(errors) > 0:
			self.config_errors = errors
			raise Exception(
				'Error(s) in configuration dictionary, in order to correct them, check config_errors attribute'
			)
		dfs = self.check_datasets(test, time_column, )
		huge_df = pd.concat(dfs, keys=range(len(dfs)))
		transf_column_list = list()
		prev_trans_dict = dict()
		for column in tqdm.tqdm(huge_df.columns):
			transf_list = list()
			grouped = huge_df[[column]].groupby(level=0, as_index=False)
			column_df = self.get_column_df_from_groups(grouped)

			for featurizer in config:
				featurizer = self.set_test_arguments(featurizer, self._featurized_data[column], column)
				transf, prev_trans = self._featurize_each_class(column_df, featurizer, use_dask, n_jobs)
				transf_list.append(transf)
				prev_trans_dict[prev_trans[0].__name__] = prev_trans[1]
			self._prev_trans_test[column] = prev_trans_dict
			transf_column_list.append(pd.concat(transf_list, axis=1, keys=[f.name for f in config]))

		self._testing_data = pd.concat(transf_column_list, axis=1, keys=huge_df.columns.values)
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
