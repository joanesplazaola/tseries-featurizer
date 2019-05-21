from ts_featurizer.tools import has_keys, function_exists, featurizer_exists, get_attr_from_module
import copy


class Execution:

	def __init__(self, name, args, kwargs, all_data, test, model):
		self.name = name
		self._args = args
		self._kwargs = kwargs
		self._send_all_data = all_data
		self.test = test
		self._model = model

	def __str__(self):
		str_list = list()
		str_list.append("_".join(map(str, self._args)))
		# str_list.append(",".join(f'{k}={v}' for k, v in self._kwargs.items()))
		if self._args:
			args = '_{}'.format("_".join(map(str, self._args)))
		else:
			args = ''
		return f'{self.name}{args}'

	@property
	def args(self):
		return self._args

	@args.setter
	def args(self, args):
		if not isinstance(args, list):
			raise TypeError('Args must be a list.')
		self._args = args

	@property
	def model(self):
		return self._model

	@model.setter
	def model(self, model):
		if not isinstance(model, dict):
			raise TypeError('Kwargs must be a dict.')
		self._model = model

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

	def __init__(self, name, class_, prev_trans, feature_confs, test=False):
		self.name = name
		self.class_ = class_
		self.prev_trans = prev_trans
		self.feature_confs = feature_confs
		self.test = test
		self.train_prev_data = None

	def get_test_features(self):
		return sum([feature_conf.get_test_execs() for feature_conf in self.feature_confs], [])

	def __str__(self):
		return self.name


def get_executions(name, executions):
	"""
	Converts a dictionary with executions, to a list of Execution objects.
	:param name:
	:param executions:
	:return:
	"""
	execs = list()
	errors = []
	if not executions:  # Check if is empty
		executions.append({})
	for execution in executions:

		if not set(execution.keys()) <= {'send_all_data', 'args', 'kwargs', 'test', 'model'}:
			errors.append(f"Only args can be: 'send_all_data', 'args', 'kwargs', 'test', 'model:  {execution.keys()}")
		args = execution.get('args', [])
		kwargs = execution.get('kwargs', {})
		all_data = execution.get('send_all_data', False)
		test = execution.get('test', {})
		model = execution.get('model', {})
		execs.append(Execution(name, args, kwargs, all_data, test, model))
	return execs, errors


def get_featurizers_errors(key, value, gen_class):
	cond_list = [
		not isinstance(key, str), not isinstance(value, dict), not has_keys({'class', 'features'}, value),
		not next(gen_class), ]
	error_list = [
		f'Featurizers identifier must be an str, {type(key)} was found',
		f'Featurizer must be a dict, {type(value)} was found.',
		'Featurizers dict must have class and features keys.',
		'Featurizers class must exist and must be BaseFeaturizers subclass.'
	]

	return [error for cond, error in zip(cond_list, error_list) if cond]


def get_previous_trans_errors(value):
	"""
	Checks the errors of the previous transformation configuration.
	:param value:
	:return:
	"""
	errors = []
	for func_name, execution in value['previous_trans'].items():
		execs, error = get_executions(func_name, [execution])
		errors.extend(error)
		value['previous_trans'][func_name] = execs[0]
	return errors


def config_validation(def_conf, test=False):
	"""
	This function checks the validity of the configuration dictionary and fills some optional gaps
	in order all the featurizers to have the same structure.
	:param def_conf: configuration dictionary
	:param test:
	:return:
	"""
	conf = copy.deepcopy(def_conf)
	featurizer_conf_list = list()
	errors = []
	for key, value in conf.configuration.items():
		gen_class = featurizer_exists(value['class'])

		errors.extend(get_featurizers_errors(key, value, gen_class))
		value.setdefault('previous_trans', {})

		errors.extend(get_previous_trans_errors(value))

		feature_list = []
		for feature_func, executions in value['features'].items():
			is_func, correct_path = function_exists(feature_func, conf.feature_paths)

			cond_list = [not isinstance(feature_func, str), not is_func]
			error_list = [
				f'Feature function must be str, {feature_func} was found.',
				f'Function not found: {feature_func}'
			]
			errors.extend([error for cond, error in zip(cond_list, error_list) if cond])

			execs, error = get_executions(feature_func, executions)
			errors.extend(error)
			func = get_attr_from_module(f'{correct_path}.{feature_func}')

			feature = FeatureConfig(feature_func, func, execs)
			feature_list.append(feature)

		featurizer_conf_list.append(FeaturizerConfig(key, next(gen_class), value['previous_trans'], feature_list, test))

	return featurizer_conf_list, errors


def get_errors(configuration):
	return config_validation(configuration)[1]
