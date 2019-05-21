import unittest
import numpy as np
import pandas as pd
from ts_featurizer.tools import get_attr_from_module, parallel_process, ARUtils
from ts_featurizer.tools import to_frequency
from ts_featurizer.tools import item_list_from_tuple_list, get_one_from_col, get_list_from_columns, \
	get_evaluated_function

from ts_featurizer.tools import has_keys, function_exists, featurizer_exists
from ts_featurizer.tools import config_dict_validation


class HelpersTest(unittest.TestCase):
	def test_get_attr_from_module_with_valid_str(self):
		function_name = 'numpy.mean'
		self.assertEqual(get_attr_from_module(function_name), np.mean, )

	def test_get_attr_from_module_with_invalid_str(self):
		function_name = 'numpy.moan'
		with self.assertRaises(AttributeError):
			get_attr_from_module(function_name)

	def test_get_attr_from_module_with_function(self):
		self.assertEqual(get_attr_from_module(np.mean), np.mean, )

	def test_get_attr_from_module_with_list(self):
		modules = ['numpy.mean', 'numpy.max']
		with self.assertRaises(TypeError):
			get_attr_from_module(modules)

	def test_get_possible_orders_valid(self):
		expected = np.array([
			[0, 0, 0],
			[0, 0, 1],
			[0, 1, 0],
			[0, 1, 1],
			[1, 0, 0],
			[1, 0, 1],
			[1, 1, 0],
			[1, 1, 1],
			[2, 0, 0],
			[2, 0, 1],
			[2, 1, 0],
			[3, 0, 0]
		])
		output = ARUtils.get_possible_orders([3, 1, 1], 3)
		self.assertTrue(np.allclose(expected, output))

	def test_get_possible_orders_negative_sum(self):
		with self.assertRaises(ValueError):
			ARUtils.get_possible_orders([3, 1, 5], -1)

	def test_get_possible_orders_negative_range(self):
		with self.assertRaises(ValueError):
			ARUtils.get_possible_orders([3, 1, -5], 3)

	def test_format_ar_column(self):
		max = 3
		params = [0.1, 0.2, 0.3]
		expected = {
			'title_0': 0.1, 'title_1': 0.2, 'title_2': 0.3
		}
		output = ARUtils.format_ar_column(params, 'title', max)

		self.assertEqual(expected, output)

	def test_function_exists(self):
		output = function_exists('numpy.mean')
		is_function = next(output)

		self.assertTrue(is_function)

	def test_function_exists_false(self):
		output = function_exists('numpy.moan')
		is_function = next(output)

		self.assertFalse(is_function)

	def test_featurizer_exists(self):
		output = featurizer_exists('ts_featurizer.tools.featurizers.TimeFeaturizer')
		is_function = next(output)

		self.assertTrue(is_function)

	def test_featurizer_exists_false(self):
		output = featurizer_exists('ts_featurizer.tools.featurizers.OnionFeaturizer')
		is_function = next(output)

		self.assertFalse(is_function)

	def test_has_keys(self):
		input = {'these': [], 'keys': []}
		output = has_keys({'these', 'keys'}, input)

		self.assertTrue(output)

	def test_has_keys_false(self):
		input = {'these': [], 'keys': []}
		output = has_keys({'not', 'these', 'keys'}, input)

		self.assertFalse(output)

	def test_parallel_process_array_linear(self):
		array = [1, 2, 3, 4, 5, 6] * 200
		expected = [1, 4, 9, 16, 25, 36] * 200
		func = lambda x: pow(x, 2)
		self.assertEqual(expected, parallel_process(array, func, n_jobs=1))


class TransformerTest(unittest.TestCase):

	def test_to_frequency_transformer_one_freq(self):
		n = 500
		freq = 2
		amplitude = 10
		expected = np.zeros(int(n / 2))
		expected[freq] = (n * amplitude) / 2
		signal = np.sin(freq * np.pi * np.arange(n) / float(n / 2)) * amplitude
		data = pd.DataFrame()
		data['signal'] = signal
		data['time'] = data.index
		data['time'] = pd.to_datetime(data['time'], unit='s')
		data.set_index('time', inplace=True)

		data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)
		frequencies = to_frequency(data)

		self.assertTrue(np.allclose(expected, frequencies['signal'].values))

	def test_to_frequency_transformer_many_freqs(self):
		N = 500
		freqs = [2, 10, 15]
		amps = [10, 20, 30]
		expected = np.zeros(int(N / 2))
		signal = np.zeros(N)
		for freq, amp in zip(freqs, amps):
			expected[freq] = (N * amp) / 2
			signal += np.sin(freq * np.pi * np.arange(N) / float(N / 2)) * amp

		data = pd.DataFrame()
		data['signal'] = signal
		data['time'] = data.index
		data['time'] = pd.to_datetime(data['time'], unit='s')
		data.set_index('time', inplace=True)

		data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)
		frequencies = to_frequency(data)

		self.assertTrue(np.allclose(expected, frequencies['signal'].values))

	def test_to_frequency_transformer_no_pandas(self):
		signal = np.arange(500)
		with self.assertRaises(AttributeError):
			to_frequency(signal)


class PreprocessorTest(unittest.TestCase):

	def test_get_frequencies_one(self):
		data = [[(555, 111), (555, 111), (555, 111), (555, 111), (555, 111)], ]
		expected = [555] * len(data[0])
		df = pd.Series(data)
		output = item_list_from_tuple_list(df, sort=False)
		self.assertListEqual(sorted(expected), sorted(output))

	def test_get_one_from_col(self):
		data = {'sensor1': [555, 555, 555, 555, 555, 555, 555, 555, 555]}
		df = pd.DataFrame(data)
		output = get_one_from_col(df, 'sensor1')
		self.assertEqual(555, output)

	def test_get_list_from_columns(self):
		data = {'func_0.5': [],
				'func_15': [],
				'func_35': [],
				'func_78': [],
				'func_0.66': [], }
		expected = ['0.5', '15', '78', '35', '0.66']
		df = pd.DataFrame(data)
		output = get_list_from_columns(df)

		self.assertListEqual(sorted(expected), sorted(output))

	def test_get_evaluated_fn_min(self):
		data = {'score': [0.5, 0.8, 0.99, 99, 108],
				'value': ['best', 'not', 'not', 'not', 'worst']}

		df = pd.DataFrame(data)
		output = get_evaluated_function(df, 'score', 'numpy.argmin', 'value')

		self.assertEqual('best', output)

	def test_get_evaluated_fn_max(self):
		data = {'score': [0.5, 0.8, 0.99, 99, 108],
				'value': ['best', 'not', 'not', 'not', 'worst']}

		df = pd.DataFrame(data)
		output = get_evaluated_function(df, 'score', 'numpy.argmax', 'value')

		self.assertEqual('worst', output)


class ConfigFactoryTest(unittest.TestCase):

	def setUp(self):
		self.valid_conf = {
			'Time': {

				'class': 'ts_featurizer.tools.featurizers.FrequencyFeaturizer',

				'previous_trans': {
					'ts_featurizer.tools.to_frequency': {'args': [], 'kwargs': {}}
				},
				'features': {'numpy.mean': [], },

			},
		}

	def test_config_dict_validation_no_errors(self):
		_, errors = config_dict_validation(self.valid_conf)
		self.assertEqual(len(errors), 0)
