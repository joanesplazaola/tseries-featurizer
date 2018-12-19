import importlib as imp
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_attr_from_module(string):
	if callable(string):
		return string
	module_name, func_name = string.rsplit('.', 1)
	module = imp.import_module(module_name)
	func = getattr(module, func_name)
	return func


def number_of_partitions(max_range, max_sum):
	'''
	Returns an array arr of the same shape as max_range, where
	arr[j] = number of admissible partitions for
			 j summands bounded by max_range[j:] and with sum <= max_sum
	'''
	m = max_sum + 1
	n = len(max_range)
	arr = np.zeros(shape=(m, n), dtype=int)
	arr[:, -1] = np.where(np.arange(m) <= min(max_range[-1], max_sum), 1, 0)
	for i in range(n - 2, -1, -1):
		for j in range(max_range[i] + 1):
			arr[j:, i] += arr[:m - j, i + 1]
	return arr.sum(axis=0)


def get_possible_orders(max_range, max_sum, out=None, n_part=None):
	if out is None:
		max_range = np.asarray(max_range, dtype=int).ravel()
		n_part = number_of_partitions(max_range, max_sum)
		out = np.zeros(shape=(n_part[0], max_range.size), dtype=int)

	if max_range.size == 1:
		out[:] = np.arange(min(max_range[0], max_sum) + 1, dtype=int).reshape(-1, 1)
		return out

	P = get_possible_orders(max_range[1:], max_sum, out=out[:n_part[1], 1:], n_part=n_part[1:])
	# P is now a useful reference

	S = np.minimum(max_sum - P.sum(axis=1), max_range[0])
	offset, sz = 0, S.size
	out[:sz, 0] = 0
	for i in range(1, max_range[0] + 1):
		ind, = np.nonzero(S)
		offset, sz = offset + sz, ind.size
		out[offset:offset + sz, 0] = i
		out[offset:offset + sz, 1:] = P[ind]
		S[ind] -= 1
	return out


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
	"""
		A parallel version of the map function with a progress bar.

		Args:
			array (array-like): An array to iterate over.
			function (function): A python function to apply to the elements of array
			n_jobs (int, default=16): The number of cores to use
			use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
				keyword arguments to function
			front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
				Useful for catching bugs
		Returns:
			[function(array[0]), function(array[1]), ...]
	"""

	# We run the first few iterations serially to catch bugs
	if front_num > 0:
		front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
	# If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
	if n_jobs == 1:
		return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
	# Assemble the workers
	with ProcessPoolExecutor(max_workers=n_jobs) as pool:
		# Pass the elements of array into function
		if use_kwargs:
			futures = [pool.submit(function, **a) for a in array[front_num:]]
		else:
			futures = [pool.submit(function, a) for a in array[front_num:]]
		kwargs = {
			'total': len(futures),
			'unit': 'it',
			'unit_scale': True,
			'leave': True
		}
		# Print out the progress as tasks complete
		for f in tqdm(as_completed(futures), **kwargs):
			pass
	out = []
	# Get the results from the futures.
	for i, future in tqdm(enumerate(futures)):
		try:
			out.append(future.result())
		except Exception as e:
			out.append(e)
	return front + out
