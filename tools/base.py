import pandas as pd
from .helpers import get_attr_from_module, parallel_process


class BaseFeaturizer:

	def __init__(self, data, config=None):
		self._data = data.copy()
		self._config = config
		self.apply_previous_trans()

	def apply_previous_trans(self):
		"""

		:return:
		"""
		transf_funcs = self._config.prev_trans

		for function_name in transf_funcs.keys():
			func = get_attr_from_module(function_name)
			args = transf_funcs[function_name].args
			kwargs = transf_funcs[function_name].kwargs
			self._data = func(self._data, *args, **kwargs)

	# TODO Ahal dan heinian hau aggr-ekin jarri biharko litzake. Holako zerbait, baina argumentuak gehituta
	#  t_list = self._data.agg(self._config.keys(), axis=1).T
	#
	def featurize(self, use_dask, n_jobs=1):
		"""

		:param use_dask:
		:param n_jobs:
		:return:
		"""

		kwargs = [
			{'key': key.name, 'exec': execs}
			for index, key in enumerate(self._config.feature_confs)
			for execs in self._config.feature_confs[index].executions
		]
		ret = parallel_process(kwargs, self._apply_featurization, n_jobs=n_jobs, use_kwargs=True)

		transf_list, method_names = map(list, zip(*ret))

		df = pd.concat(transf_list, axis=1, )
		df.columns = method_names
		return df, self._data  # The transformed dataframe and the previous transformations dataframe are returned

	def _apply_featurization(self, key, exec):
		"""

		:param key:
		:param exec:
		:return:
		"""

		func = get_attr_from_module(key)

		args = tuple(exec.args)
		kwargs = exec.kwargs
		if exec.send_all_data:
			kwargs['all_data'] = self._data
		trans = self._data.apply(func, args=args, **kwargs)
		return trans, str(exec)
