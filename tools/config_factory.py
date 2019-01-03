from tools import BaseFeaturizer, get_attr_from_module


class Execution:

	def __init__(self, name, args, kwargs, all_data, test):
		self.name = name
		self.args = args
		self.kwargs = kwargs
		self.send_all_data = all_data
		self.test = test

	def __str__(self):
		str_list = list()
		str_list.append(",".join(map(str, self._args)))
		str_list.append(",".join(f'{k}={v}' for k, v in self._kwargs.items()))
		str_list = list(filter(None, str_list))
		return f'{self.name}({", ".join(str_list)} )'

	@property
	def args(self):
		return self._args

	@args.setter
	def args(self, args):
		if not isinstance(args, list):
			raise TypeError('Args must be a list.')
		self._args = args

	@property
	def kwargs(self):
		return self._kwargs

	@kwargs.setter
	def kwargs(self, kwargs):
		if not isinstance(kwargs, dict):
			raise TypeError('Kwargs must be a dict.')
		self._kwargs = kwargs

	@property
	def send_all_data(self):
		return self._send_all_data

	@send_all_data.setter
	def send_all_data(self, send_all_data):
		if not isinstance(send_all_data, bool):
			raise TypeError('send_all_data must be a boolean')
		self._send_all_data = send_all_data


class FeatureConfig:

	def __init__(self, name, function_, executions):
		self.name = name
		self.function_ = function_
		self.executions = executions

	def __str__(self):
		return self.function_.__name__

	def get_test_execs(self):
		"""
		Get all the executions where the 'test' is not an empty dict.
		:return: All test executions.
		"""
		return list(filter(lambda execution: execution.test, self.executions))


class FeaturizerConfig:

	def __init__(self, name, class_, prev_trans, feature_confs):
		self.name = name
		self.class_ = class_
		self.prev_trans = prev_trans
		self.feature_confs = feature_confs

	def get_test_features(self):
		return sum([feature_conf.get_test_execs() for feature_conf in self.feature_confs], [])


def featurizer_exists(module):
	"""
	Checks if class exists
	:param module: Class' string representation
	:return: (boolean) if is function
	"""
	class_ = get_attr_from_module(module)

	yield issubclass(class_, BaseFeaturizer)
	yield class_


def function_exists(feature_func):
	"""
	Checks if specified function exists
	:param feature_func: Function's string representation.
	:return: (boolean, function) if is function
	"""
	func = get_attr_from_module(feature_func)

	yield callable(func)
	yield func


def has_keys(keys, dict):
	"""
	Checks if a dictionary has specified keys.
	:param keys: Set of keys.
	:param dict: Dictionary
	:return:
	"""
	return keys <= set(dict)


def get_executions(name, executions):
	execs = list()
	errors = []
	if not executions:  # Check if is empty
		executions.append({})
	for execution in executions:

		if not set(execution.keys()) <= {'send_all_data', 'args', 'kwargs', 'test'}:
			errors.append(f"Only args can be: 'send_all_data', 'args', 'kwargs', 'test':  {execution.keys()}")
		args = execution.get('args', [])
		kwargs = execution.get('kwargs', {})
		all_data = execution.get('send_all_data', False)
		test = execution.get('test', False)
		execs.append(Execution(name, args, kwargs, all_data, test))
	return execs, errors


def config_dict_validation(def_conf):
	"""
	This function checks the validity of the configuration dictionary and fills some optional gaps
	in order all the featurizers to have the same structure.
	:param def_conf:
	:return:
	"""
	featurizer_conf_list = list()
	errors = []
	for key, value in def_conf.items():
		gen_class = featurizer_exists(value['class'])
		cond_list = [
			not isinstance(key, str), not isinstance(value, dict), not has_keys({'class', 'features'}, value),
			not next(gen_class), ]
		error_list = [
			f'Featurizers identifier must be an str, {type(key)} was found',
			f'Featurizer must be a dict, {type(value)} was found.',
			'Featurizers dict must have class and features keys.',
			'Featurizers class must exist and must be BaseFeaturizers subclass.'
		]

		errors.extend([error for cond, error in zip(cond_list, error_list) if cond])
		value.setdefault('previous_trans', {})
		for func_name, execution in value['previous_trans'].items():
			execs, error = get_executions(func_name, [execution])
			errors.extend(error)
			value['previous_trans'][func_name] = execs[0]

		feature_conf_list = []
		for feature_func, executions in value['features'].items():
			gen_func = function_exists(feature_func)
			cond_list = [not isinstance(feature_func, str), not next(gen_func)]
			error_list = [
				f'Feature function must be str, {feature_func} was found.',
				f'Function not found: {feature_func}'
			]
			errors.extend([error for cond, error in zip(cond_list, error_list) if cond])

			execs, error = get_executions(feature_func, executions)
			errors.extend(error)
			feature_conf_list.append(FeatureConfig(name=feature_func, function_=next(gen_func), executions=execs))

		featurizer_conf_list.append(FeaturizerConfig(key, next(gen_class), value['previous_trans'], feature_conf_list))

	return featurizer_conf_list, errors
