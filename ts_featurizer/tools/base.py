import pandas as pd
from .helpers import get_attr_from_module


class BaseFeaturizer:

	def __init__(self, config=None):
		self._config = config
		self._transformed_data = None

	def apply_previous_trans(self, data, test, train_prev_data=None):
		"""
		This function transforms the original (time-domain) data into new data.
		For example, the FrequencyFeaturizer needs first to be transformed from time to frequency domain.

		:param data: The data to transform
		:return: The data after applying the transformation.
		"""
		transf_funcs = self._config.prev_trans

		for function_name in transf_funcs.keys():
			func = get_attr_from_module(function_name)
			args = transf_funcs[function_name].args
			kwargs = transf_funcs[function_name].kwargs
			if test:
				kwargs['test'] = True
				kwargs['train_prev_data'] = train_prev_data
			data = func(data, *args, **kwargs)
		return data

	def featurize(self, data, ):
		"""
		This function gets the featurized data for each feature in a featurizer, calling _apply_featurization function
		with the specified parallelization.
		For example, the TimeFeaturizer has many features defined (max, min, mean...), so this function would call
		parallel_process with all the configuration set, and would get a list of the featurized data.


		Firstly, prepares the configuration, separating each execution in a dictionary, storing it inside a list.

		Each execution contains the args, kwargs and other settings for the featurization to complete, and the key is
		the function that will be applied to the previously transformed dataset.

		After the configuration is set, the parallel_process function is called, where _apply_featurization function
		will be parallelly called with the specified configuration and n_jobs.

		This returns a list of dfs with the featurized data, and finally this is concatenated to a unique df.
		:param data: The data to featurize
		:param use_dask: Specifies if the library for parallel computing is used. (Not implemented)
		:param n_jobs: Specifies the number of cores used for the parallelization.
		:return: The featurized data df and the previous transformations df are returned
		"""

		self._transformed_data = self.apply_previous_trans(data.copy(), self._config.test, self._config.train_prev_data)

		featurized_list = list()
		for index, key in enumerate(self._config.feature_confs):
			for execs in self._config.feature_confs[index].executions:
				featurized_list.append(self._apply_featurization(function=key.function_, executions=execs))

		featurized_df = pd.concat(featurized_list, axis=1, )
		return featurized_df, self._transformed_data

	@staticmethod
	def _featurize_all_columns(trans_data, func, args, kwargs):
		trans_list = list()

		for col in trans_data:
			column = trans_data[col].dropna()
			value = func(column, *args, **kwargs)
			trans_list.append(value)

		trans = pd.Series(trans_list)
		return trans

	def _apply_featurization(self, function, executions):
		"""
		Applies the featurization function to the data.

		As sometimes data needs to be compared between columns, all_data parameter
		is used, to know whether all the dataset needs to be sent for each call.

		The result got from apply (trans) has to be a Series of floats or Series of dicts with float values.

		Complete gaps is implemented in order to improve the uniformity of the returned Series of dicts.
		For example in case of the frequency, peaks are not always found in the same frequencies in different
		signals, so, the detect peaks is only going to return the amplitude of the peak frequencies, getting many NaNs.

		After getting all the peak frequencies of all the signals, the apply function is called again,
		with the sum of all found peak frequencies, getting the amplitudes of each signal in those peak frequencies
		and in this way, a better uniformity is achieved.

		If trans is a Series of dicts, the keys are added to the name of the feature, in order to difference them.
		For example, if a peak has been found in 4, 15 and 68 Hz in frequency, we will have three columns:

		tools.detect_peaks_4 | tools.detect_peaks_15 | tools.detect_peaks_68

		If trans is just a Series of floats, the name of the function will be the name of the only column of the df:
		For example, when we are calculating the max of each column of a df, we will just get a Series of floats after
		applying the function, so we will have one column called as the name of the function applied, in this case:

		numpy.max

		:param key: Name of the function to apply
		:param executions: Execution data, with args, kwargs and other configuration settings.
		:return: a pd.DataFrame with the featurized data.
		"""

		args = executions.args
		kwargs = executions.kwargs
		if executions.send_all_data:
			kwargs['all_data'] = self._transformed_data

		if executions.model:
			func_model = get_attr_from_module(executions.model['preprocessor'])
			model = func_model(self._transformed_data, **executions.model['kwargs'])
			kwargs['fit_model'] = model
			kwargs['test'] = True

		trans = self._featurize_all_columns(self._transformed_data, function, args, kwargs)

		df = pd.DataFrame(trans.tolist())

		if isinstance(trans[0], dict):
			df.columns = map(lambda key: f'{executions}_{key}', df.columns)
		else:
			df.columns = [str(executions)]

		return df
