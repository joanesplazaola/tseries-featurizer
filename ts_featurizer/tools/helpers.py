import importlib as imp
import numpy as np
import numba as nb


@nb.jit
def _roll(a, shift):
	"""
	Roll 1D array elements. Improves the performance of numpy.roll() by reducing the overhead introduced from the
	flexibility of the numpy.roll() method such as the support for rolling over multiple dimensions.

	Elements that roll beyond the last position are re-introduced at the beginning. Similarly, elements that roll
	back beyond the first position are re-introduced at the end (with negative shift).

	Examples
	--------
	>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	>>> _roll(x, shift=2)
	>>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

	>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	>>> _roll(x, shift=-2)
	>>> array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

	>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	>>> _roll(x, shift=12)
	>>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

	Benchmark
	---------
	>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	>>> %timeit _roll(x, shift=2)
	>>> 1.89 µs ± 341 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

	>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	>>> %timeit np.roll(x, shift=2)
	>>> 11.4 µs ± 776 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

	:param a: the input array
	:type a: array_like
	:param shift: the number of places by which elements are shifted
	:type shift: int
	:return: shifted array with the same shape as a
	:return type: ndarray
	"""

	idx = shift % len(a)
	return np.concatenate([a[-idx:], a[:-idx]])


def get_attr_from_module(module):
	"""
	Returns the attribute of a module.
	For example, for 'numpy.max' the max function is returned.
	:param module: string containing the dir of the function/class.
	:return: specified class or function.
	"""
	if callable(module):
		return module
	elif not isinstance(module, str):
		raise TypeError('Argument must be string.')
	module_name, func_name = module.rsplit('.', 1)
	module = imp.import_module(module_name)
	func = getattr(module, func_name)
	return func


def featurizer_exists(module):
	"""
	Checks if class exists
	:param module: Class' string representation
	:return: (boolean) if is function
	"""
	from .base import BaseFeaturizer

	try:
		class_ = get_attr_from_module(module)
	except AttributeError:
		class_ = type(None)
	yield issubclass(class_, BaseFeaturizer)
	yield class_


def function_exists(feature_func, paths):
	"""
	Checks if specified function exists
	:param feature_func: Function's string representation.
	:return: (boolean, function) if is function
	"""

	for path in paths:

		try:
			func = get_attr_from_module(f'{path}.{feature_func}')
			correct_path = path
		except AttributeError:
			continue
		return callable(func), correct_path
	return False, ''


def has_keys(keys, dict):
	"""
	Checks if a dictionary has specified keys.
	:param keys: Set of keys.
	:param dict: Dictionary
	:return:
	"""
	return keys <= set(dict)


class ARUtils:
	@staticmethod
	def get_orders_aic(order, data):
		"""
		This function creates an ARIMA model with specified orders, and fits it.
		After training the model, the AIC and the parameters are returned.
		:param order:
		:param data:
		:return:
		"""
		import warnings
		from statsmodels.tsa.arima_model import ARIMA
		import numpy as np

		aic = float('inf')
		data.name = str(data.name)
		p, q, c = None, None, None
		try:
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore")
				arima_mod = ARIMA(data, order).fit(disp=0)
				aic = arima_mod.aic
				p, q, c = arima_mod.arparams, arima_mod.maparams, arima_mod.params.const
		except (ValueError, np.linalg.LinAlgError) as e:

			pass

		return aic, (p, q, [c])

	@staticmethod
	def get_best_order(data, max_coeffs):
		"""
		This function returns the order with the best AIC, taking the max_coeffs of the model into account.
		:param data:
		:param max_coeffs:
		:return:
		"""
		best_score, best_cfg = float("inf"), None
		orders = ARUtils.get_possible_orders([max_coeffs, max_coeffs], max_coeffs)
		orders = np.concatenate((np.insert(orders, 1, 1, axis=1), np.insert(orders, 1, 0, axis=1)), )

		for order in orders:
			aic, _ = ARUtils.get_orders_aic(order, data)
			if aic < best_score:
				best_score, best_cfg = aic, order
		return best_cfg

	@staticmethod
	def format_ar_column(params, title, max):
		"""
		This method formats the AR column for the returning Dataframe.
		Example:
		{'ar_params_0': 0.55, 'ar_params_1': 0.32, 'ar_params_2':0.785}
		:param params:
		:param title:
		:param max:
		:return:
		"""
		if params is None:
			params = [None] * max
		return {f'{title}_{index}': params[index] for index in range(max)}

	@staticmethod
	def get_possible_orders(max_ranges, max_sum):
		"""
		Calculates the possible orders with max_range that in summation give less than max_sum.
		For example:

		get_possible_orders([3, 3], 2)

		Out[1]:
			array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0]])

		:param max_range: Max range of each element of array.
		:param max_sum: Max sum of all the elements in the order.
		:return: array of possible orders.
		"""
		if max_sum <= 0:
			raise ValueError('Max_sum must be a positive number.')

		if any(item < 0 for item in max_ranges):
			raise ValueError('All elements in max_range list must be bigger or equal to zero.')
		range_list = [range(max_range + 1) for max_range in max_ranges]

		orders = np.stack(np.meshgrid(*range_list), -1).reshape(-1, len(max_ranges))
		mask = np.where(np.sum(orders, axis=1) <= max_sum)
		return orders[mask]
