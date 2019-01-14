import unittest
import numpy as np
import pandas as pd
from tools import get_attr_from_module, get_possible_orders, parallel_process
from tools import to_frequency
from tools import item_list_from_tuple_list


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
		output = get_possible_orders([3, 1, 1], 3)
		self.assertTrue(np.allclose(expected, output))

	def test_get_possible_orders_negative_sum(self):
		with self.assertRaises(ValueError):
			get_possible_orders([3, 1, 5], -1)

	def test_get_possible_orders_negative_range(self):
		with self.assertRaises(ValueError):
			get_possible_orders([3, 1, -5], 3)

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
