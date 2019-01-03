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
		return list(filter(lambda execution: execution.test, self.executions))


class FeaturizerConfig:

	def __init__(self, name, class_, prev_trans):
		self.name = name
		self.class_ = class_
		self.prev_trans = prev_trans
		self.feature_confs = list()

	def get_test_features(self):
		return sum([feature_conf.get_test_execs() for feature_conf in self.feature_confs], [])


def featurizer_exists(module):
	class_ = get_attr_from_module(module)

	return issubclass(class_, BaseFeaturizer), class_


def feature_function_exists(feature_func):
	func = get_attr_from_module(feature_func)

	return callable(func), func


def dict_has_keys(keys, dict):
	"""

	:param keys:
	:param dict:
	:return:
	"""
	return keys <= set(dict)


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
		if not isinstance(key, str):
			errors.append(f'Featurizers identifier must be an str, {type(key)} was found')
		if not isinstance(value, dict):
			errors.append(f'Featurizer must be a dict, {type(key)} was found.')
		if not dict_has_keys({'class', 'features'}, value):
			errors.append(f'Featurizers dict must have class and features keys.')
		exists, class_ = featurizer_exists(value['class'])
		if not exists:
			errors.append(f'Featurizers class must exist and must be BaseFeaturizers subclass.')
		value.setdefault('previous_trans', {})

		for func_name, args in value['previous_trans'].items():
			if not set(args.keys()) <= {'send_all_data', 'args', 'kwargs', 'test'}:
				errors.append(
					f"Only args can be: 'send_all_data', 'args', 'kwargs', 'test':  {args.keys()}")

			value['previous_trans'][func_name] = Execution(func_name, args.get('args', []), args.get('kwargs', {}),
														   args.get('send_all_data', False), args.get('test', {}))
		feature_conf_list = list()
		for feature_func, arg in value['features'].items():

			if not isinstance(feature_func, str):
				errors.append(f'Feature function must be str, {feature_func} was found.')
			exists, func = feature_function_exists(feature_func)

			if not exists:
				errors.append(f'Function not found: {feature_func}')
				break
			if not arg:  # Check if is empty
				arg.append({})
			execs = list()
			for args in arg:

				# Check if there is any key other than send_all_data, args and kwargs
				if not set(args.keys()) <= {'send_all_data', 'args', 'kwargs', 'test'}:
					errors.append(
						f"Only args can be: 'send_all_data', 'args', 'kwargs', 'test':  {args.keys()}")

				execs.append(Execution(feature_func, args.get('args', []), args.get('kwargs', {}),
									   args.get('send_all_data', False), args.get('test', {})))
			feature_conf_list.append(FeatureConfig(name=feature_func, function_=func, executions=execs))

		f_config = FeaturizerConfig(name=key, class_=class_, prev_trans=value['previous_trans'])
		f_config.feature_confs = feature_conf_list
		featurizer_conf_list.append(f_config)

	return errors, featurizer_conf_list
