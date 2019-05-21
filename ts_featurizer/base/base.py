import pandas as pd
import numpy as np
import copy
from multiprocessing import Pool, cpu_count
from ts_featurizer.tools import get_attr_from_module
from ts_featurizer.config import config_validation, DefaultConfiguration
from tqdm import tqdm
import warnings

formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
	formatwarning_orig(message, category, filename, lineno, line='')


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

	constant_columns: list
		Columns that do not change with the time are stored here and not computed, as they don't provide info.

	_collapse_columns: bool
		Boolean that specifies if the created multi-index array's columns must be collapsed.

	_featurize_constant_cols: bool
		Boolean that specifies if those columns that are considered constants have to be used as output features.

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

	def __init__(self, config=DefaultConfiguration(), collapse_columns=True, featurize_ct_cols=True, check_na=True):
		self._collapse_columns = collapse_columns
		self._featurized_data = pd.DataFrame()
		self._testing_data = pd.DataFrame()
		self._prev_trans = dict()
		self.na_cols = dict()
		self.constant_columns = list()
		self._featurize_constant_cols = featurize_ct_cols
		self._raw_config = copy.deepcopy(config)
		self._model_ready = False
		self._check_na = check_na

	def featurize(self, train, time_column=None, use_dask=False, n_jobs=-1, apply_model=False):
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

		The way or working changes slightly when apply_model is True

		:param train: dataFrame used for the training. Featurization is applied to this data.
		:param time_column: The date column in order the time series to be ordered properly.
		:param use_dask: Specifies if the library for parallel computing is used. (Not implemented)
		:param n_jobs: Specifies the number of cores used for the parallelization.
		:param apply_model: Specifies if have to create the model or apply it.
		:return: featurized DataFrame.

		"""

		application_mode = "Applying the model" if apply_model else "Modeling"

		self._config, errors = config_validation(self._raw_config, test=apply_model)

		if len(errors) > 0:
			self.config_errors = errors
			raise Exception(
				'Error(s) in configuration dictionary, in order to correct them, check config_errors attribute'
			)
		elif apply_model and not self._model_ready:
			raise Exception(
				'In order to use apply_model mode, firstly a model has to be created.'
				' This is done by calling the function without the apply_model parameter.'
			)

		if n_jobs < 1:
			n_jobs = 2 * cpu_count()

		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("once")
			dfs = self.check_datasets(train, time_column)
		huge_df = pd.concat(dfs, keys=list(range(len(dfs))))

		huge_df, self.constant_columns, featurized_constants = self.filter_constant_values(huge_df, test=apply_model,
																						   dfs=dfs)

		kwargs = [{
			'column': column,
			'config': [
				self.set_test_arguments(featurizer, self._featurized_data[column], column)
				for featurizer in self._config
			] if apply_model else self._config,
			'col_df': huge_df[[column]]
		} for column in huge_df.columns
		]
		print(f'\n\n{"-" * 50} {application_mode} started {"-" * 50}')
		with Pool(n_jobs) as p:
			ret = list(tqdm(p.imap(self.get_featurized_col, kwargs), total=len(kwargs)))
		transf_column_list, prev_trans = map(list, zip(*ret))

		if apply_model:
			testing_data = pd.concat(transf_column_list, axis=1)
			trans_copy = self.add_constants(featurized_constants, testing_data)

		else:
			self._prev_trans = {k: v for element in prev_trans for k, v in element.items()}
			self._featurized_data = pd.concat(transf_column_list, axis=1)
			self._featurized_data = self.add_constants(featurized_constants, self._featurized_data)
			trans_copy = self._featurized_data.copy(deep=True)
			self._model_ready = True

		self.na_cols[application_mode] = trans_copy.columns[trans_copy.isna().any()].tolist()
		if len(self.na_cols[application_mode]) > 0:
			warnings.warn(
				f'NaN values have been found while {application_mode}, check na_cols attribute to know which columns have NaNs.')

		if self._collapse_columns:
			trans_copy.columns = trans_copy.columns.to_series().str.join('_')

		return trans_copy

	def add_constants(self, featurized_constants, feat_data):
		if self._featurize_constant_cols:
			featurized_constants = pd.concat([featurized_constants], keys=['Values'], axis=1)
			featurized_constants = pd.concat([featurized_constants], keys=['Constant'], axis=1)
			feat_data = pd.concat([feat_data, featurized_constants], axis=1)
		return feat_data

	def filter_constant_values(self, huge_df, test, dfs=None):
		constant_feat = pd.DataFrame()
		if not test:
			std_df = pd.DataFrame()
			for idx, df in enumerate(dfs):
				std_df[str(idx)] = df.std()

			constant_cols = huge_df.columns[std_df.T.sum() < 0.000000001].values
		else:

			# When testing, those columns filtered in training are used
			constant_cols = self.constant_columns

		if self._featurize_constant_cols:

			for idx, df in enumerate(dfs):
				constant_feat[idx] = df[constant_cols].mode().iloc[0]

		constant_feat = constant_feat.T

		# This filters the columns where values are constant
		huge_df = huge_df.loc[:, ~huge_df.columns.isin(constant_cols)]
		return huge_df, constant_cols, constant_feat

	@staticmethod
	def _check_dataset(data, time_column, check_na):
		"""
		Dataset validation is performed in this function, checking NaNs, data type, index's validity...


		:param data: dataFrame to be validated.
		:param time_column: Date column of the DataFrame.
		:return: validated data, with some changes in index, etc.
		"""
		if not isinstance(data, pd.DataFrame):
			raise TypeError('Data must be a pandas DataFrame.')

		if data.isna().sum().sum() > 0 and check_na:
			raise Exception('NaNs have been found in the data.')
		if not time_column:
			if not data.index.is_all_dates:
				try:
					date_typed = pd.to_datetime(np.array(range(len(data.index.values))), unit='s')
					data.set_index(date_typed, inplace=True)

					warnings.warn('Index was used as the time reference column.')

				except Exception as e:
					raise Exception('No time column was specified, and DataFrame\'s index values are not dates.')

		else:

			if time_column not in data.select_dtypes(include=[np.datetime64, ]).columns:
				try:
					data[time_column] = pd.to_datetime(data[time_column], unit='s')
					warnings.warn('Time column was converted from numeric to date type.')

				except Exception as e:
					raise Exception('Specified time column is not of datetime type.')

			data.set_index(time_column, inplace=True)

		data = data.select_dtypes(include=[np.number])
		data.sort_index(inplace=True)
		data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)

		return data

	@staticmethod
	def check_all_same_columns(df_list):
		"""
		Checks if in a list of pd.DataFrame all DataFrame s have the same columns.
		:param df_list: list of pd.Dataframes to check.
		:return: Whether all the DataFrames have same columns.
		"""
		cols = df_list[0].columns.tolist()

		return all(x.columns.tolist() == cols for x in df_list)

	def check_datasets(self, dfs, time_column):
		"""
		Validates and formats list of dataframes.

		:param df: list of DataFrames or DataFrame
		:param time_column:
		:return: formatted list of DataFrames
		"""
		df_list = list()

		if isinstance(dfs, pd.DataFrame):
			dfs = [dfs]
		if not self.check_all_same_columns(dfs):
			raise Exception('Dataframes in list are not comparable as have different columns.')

		for df in dfs:
			df_list.append(self._check_dataset(df.copy(deep=True), time_column, self._check_na))

		return df_list

	@staticmethod
	def get_featurized_col(kwargs):

		column = kwargs['column']
		config = kwargs['config']
		col_df = kwargs['col_df']

		column_df = TimeSeriesFeaturizer.get_column_df_from_groups(col_df.groupby(level=0))
		transf_list = list()
		prev_trans_dict = dict()
		for featurizer in config:
			transf, prev_trans = TimeSeriesFeaturizer.featurize_each_class(column_df, featurizer)
			transf_list.append(transf)
			if len(featurizer.get_test_features()) > 0:
				prev_trans_dict[prev_trans[0].__name__] = prev_trans[1]
		featurized_col = pd.concat(transf_list, axis=1, keys=[f.name for f in config])
		featurized_col = pd.concat([featurized_col], axis=1, keys=[column])
		return featurized_col, {column: prev_trans_dict}

	def set_test_arguments(self, feat, f_data, column):

		featurizer = copy.deepcopy(feat)
		test_executions = featurizer.get_test_features()
		if len(featurizer.get_test_features()) > 0:
			featurizer.train_prev_data = self._prev_trans[column][featurizer.class_.__name__]
		for execution in test_executions:
			ret = f_data[featurizer.name].loc[:, f_data[featurizer.name].columns.str.startswith(str(execution))]
			ret = self.apply_functions_to_data(execution.test['preprocessors'], ret)
			execution.kwargs['fit_model'] = ret
			execution.kwargs['test'] = True
			execution.model = {}

		return featurizer

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

	@staticmethod
	def featurize_each_class(data, config):
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
		transf, prev_transf = featurizer.featurize(data, )

		return transf, (config.class_, prev_transf)

	@staticmethod
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
