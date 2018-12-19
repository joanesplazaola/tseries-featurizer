import pandas as pd
from .helpers import get_attr_from_module, parallel_process


class BaseFeaturizer:

	def __init__(self, data, time_config=None):
		self._data = data.copy()
		self._config = time_config

		self.apply_previous_trans()

	def apply_previous_trans(self):
		transf_funcs = self._config['previous_trans']

		for function_name in transf_funcs.keys():
			func = get_attr_from_module(function_name)
			args = transf_funcs[function_name].get('args', [])
			kwargs = transf_funcs[function_name].get('kwargs', {})
			self._data = func(self._data, *args, **kwargs)

	# TODO Ahal dan heinian hau aggr-ekin jarri biharko litzake. Holako zerbait, baina argumentuak gehituta
	#  t_list = self._data.agg(self._config.keys(), axis=1).T
	#
	def featurize(self, use_dask, n_jobs=1):

		kwargs = [
			{'key': key, 'exec': execs}
			for key in self._config['features']
			for execs in self._config['features'][key]
		]
		ret = parallel_process(kwargs, self._apply_featurization, n_jobs=n_jobs, use_kwargs=True)

		transf_list, method_names = map(list, zip(*ret))

		df = pd.concat(transf_list, axis=1, )
		df.columns = method_names
		return df, self._data  # The transformed dataframe and the previous transformations dataframe are returned

	def _apply_featurization(self, key, exec):

		func = get_attr_from_module(key)

		args = tuple(exec.get('args', []))
		kwargs = exec.get('kwargs', {})
		if exec.get('send_all_data', False):
			kwargs['all_data'] = self._data
		trans = self._data.apply(func, args=args, **kwargs)
		str_args = ",".join(map(str, args))  # TODO Hemen kwarg-ekin zerbaitt eittia ondo legoke
		return trans, f'{func.__name__}{ str_args}'
