import importlib as imp
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def get_attr_from_module(module):
	if callable(module):
		return module
	elif not isinstance(module, str):
		raise TypeError('Argument must be string.')

	module_name, func_name = module.rsplit('.', 1)
	module = imp.import_module(module_name)
	func = getattr(module, func_name)
	return func


def get_possible_orders(max_range, max_sum):
	if max_sum <= 0:
		raise ValueError('Max_sum must be a positive number.')

	if any(item < 0 for item in max_range):
		raise ValueError('All elements in max_range list must be bigger or equal to zero.')
	max_range = np.asarray(max_range, dtype=int).ravel()
	if max_range.size == 1:
		return np.arange(min(max_range[0], max_sum) + 1, dtype=int).reshape(-1, 1)
	P = get_possible_orders(max_range[1:], max_sum)
	# S[i] is the largest summand we can place in front of P[i]
	S = np.minimum(max_sum - P.sum(axis=1), max_range[0])
	offset, sz = 0, S.size
	out = np.empty(shape=(sz + S.sum(), P.shape[1] + 1), dtype=int)
	out[:sz, 0] = 0
	out[:sz, 1:] = P
	for i in range(1, max_range[0] + 1):
		ind, = np.nonzero(S)
		offset, sz = offset + sz, ind.size
		out[offset:offset + sz, 0] = i
		out[offset:offset + sz, 1:] = P[ind]
		S[ind] -= 1
	return out


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
	"""
	A parallel version of the map function.

	:param array: An array to iterate over.
	:param function: A python function to apply to the elements of array
	:param n_jobs: The number of cores to use, defaults to 16.
	:param use_kwargs: Whether to consider the elements of array as dictionaries of
				keyword arguments to function
	:param front_num: The number of iterations to run serially before kicking off the parallel job.
				Useful for catching bugs
	:type array: iterable
	:type function: function
	:type n_jobs: int
	:type use_kwargs: bool
	:type front_num: int
	:return: [function(array[0]), function(array[1]), ...]
	"""
	front = []
	# We run the first few iterations serially to catch bugs
	if front_num > 0:
		front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
	# If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
	if n_jobs == 1:
		return front + [function(**a) if use_kwargs else function(a) for a in array[front_num:]]
	# Assemble the workers
	with ProcessPoolExecutor(max_workers=n_jobs) as pool:
		# Pass the elements of array into function
		if use_kwargs:
			futures = [pool.submit(function, **a) for a in array[front_num:]]
		else:
			futures = [pool.submit(function, a) for a in array[front_num:]]

	out = []
	# Get the results from the futures.
	for i, future in enumerate(futures):
		try:
			out.append(future.result())
		except Exception as e:
			out.append(e)
	return front + out
