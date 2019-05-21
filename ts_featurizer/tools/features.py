"""Detect peaks in data based on their amplitude and other features."""

from __future__ import division, print_function
import numpy as np
import pandas as pd
import numba as nb
import scipy as sp
from functools import wraps
from scipy.signal import cwt, find_peaks_cwt, ricker, welch

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.5"
__license__ = "MIT"


def to_array(func):
	@wraps(func)
	def series_to_ndarray(x, *args, **kwargs):
		return func(np.asarray(x), *args, **kwargs)

	return series_to_ndarray


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, test=False, fit_model=None):
	"""Detect peaks in data based on their amplitude and other features.

	Parameters
	----------
	x : 1D array_like
		data.
	mph : {None, number}, optional (default = None)
		detect peaks that are greater than minimum peak height (if parameter
		`valley` is False) or peaks that are smaller than maximum peak height (if parameter `valley` is True).
	mpd : positive integer, optional (default = 1)
		detect peaks that are at least separated by minimum peak distance (in
		number of data).
	threshold : positive number, optional (default = 0)
		detect peaks (valleys) that are greater (smaller) than `threshold`
		in relation to their immediate neighbors.
	edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
		for a flat peak, keep only the rising edge ('rising'), only the
		falling edge ('falling'), both edges ('both'), or don't detect a
		flat peak (None).
	kpsh : bool, optional (default = False)
		keep peaks with same height even if they are closer than `mpd`.
	valley : bool, optional (default = False)
		if True (1), detect valleys (local minima) instead of peaks.
	show : bool, optional (default = False)
		if True (1), plot data in matplotlib figure.
	ax : a matplotlib.axes.Axes instance, optional (default = None).

	Returns
	-------
	ind : 1D array_like
		indeces of the peaks in `x`.

	Notes
	-----
	The detection of valleys instead of peaks is performed internally by simply
	negating the data: `ind_valleys = detect_peaks(-x)`

	The function can handle NaN's

	See this IPython Notebook [1]_.

	References
	----------
	.. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

	Examples
	--------
	from detect_peaks import detect_peaks
	x = np.random.randn(100)
	x[60:81] = np.nan
	# detect all peaks and plot data
	ind = detect_peaks(x, show=True)
	print(ind)

	x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
	# set minimum peak height = 0 and minimum peak distance = 20
	detect_peaks(x, mph=0, mpd=20, show=True)

	x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
	# set minimum peak distance = 2
	detect_peaks(x, mpd=2, show=True)

	x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
	# detection of valleys instead of peaks
	detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

	x = [0, 1, 1, 0, 1, 1, 0]
	# detect both edges
	detect_peaks(x, edge='both', show=True)

	x = [-2, 1, -2, 2, 1, 1, 3, 0]
	# set threshold = 2
	detect_peaks(x, threshold = 2, show=True)

	Version history
	---------------
	'1.0.5':
		The sign of `mph` is inverted if parameter `valley` is True

	"""

	data = pd.Series(x)

	if test:
		return {i: round(x[i], 3) for i in fit_model}

	if mph is None:
		mph = np.percentile(x, 90)
	x = np.atleast_1d(x).astype('float64')
	if x.size < 3:
		return np.array([], dtype=int)
	if valley:
		x = -x
		if mph is not None:
			mph = -mph
	# find indices of all peaks
	dx = x[1:] - x[:-1]

	ine, ire, ife = np.array([[], [], []], dtype=int)
	if not edge:
		ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
	else:
		if edge.lower() in ['rising', 'both']:
			ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
		if edge.lower() in ['falling', 'both']:
			ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
	ind = np.unique(np.hstack((ine, ire, ife)))

	# first and last values of x cannot be peaks
	if ind.size and ind[0] == 0:
		ind = ind[1:]
	if ind.size and ind[-1] == x.size - 1:
		ind = ind[:-1]
	# remove peaks < minimum peak height
	if ind.size and mph is not None:
		ind = ind[x[ind] >= mph]
	# remove peaks - neighbors < threshold
	if ind.size and threshold > 0:
		dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
		ind = np.delete(ind, np.where(dx < threshold)[0])
	# detect small peaks closer than minimum peak distance
	if ind.size and mpd > 1:
		ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
		idel = np.zeros(ind.size, dtype=bool)
		for i in range(ind.size):
			if not idel[i]:
				# keep peaks with the same height if kpsh is True
				idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
					   & (x[ind[i]] > x[ind] if kpsh else True)
				idel[i] = 0  # Keep current peak
		# remove the small peaks and sort back the indices by their occurrence
		ind = np.sort(ind[~idel])

	ret = {data[data == data.iloc[i]].index[0]: round(data[i], 3) for i in ind}
	return ret


def get_AR_params(data, fit_model, test=False):
	from .helpers import ARUtils
	clean_data = data.dropna()
	_, params = ARUtils.get_orders_aic(fit_model, clean_data)
	item_desc = ['p', 'd', 'q']
	best_order = {f'best_order_{model_name}': model_num for model_num, model_name in zip(fit_model, item_desc)}
	ar_params = ARUtils.format_ar_column(params[0], 'ar_params', fit_model[0])
	ma_params = ARUtils.format_ar_column(params[1], 'ma_params', fit_model[2])
	ret = {'constant': params[2][0], **best_order, **ar_params, **ma_params}
	return ret


@to_array
@nb.njit
def signal_energy(data):
	"""
	Returns the absolute energy of the time series which is the sum over the squared values
	.. math::
		E = \\sum_{i=1,\ldots, n} x_i^2
	:param data: the time series to calculate the feature of
	:type data: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""
	return np.sum(data ** 2)


@to_array
@nb.njit
def sum_(data):
	return np.sum(data)


def mean_second_derivative_central(x):
	"""
	Returns the mean value of a central approximation of the second derivative
	.. math::
		\\frac{1}{n} \\sum_{i=1,\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
	from .helpers import _roll

	diff = (_roll(x, 1) - 2 * np.array(x) + _roll(x, -1)) / 2.0
	return np.mean(diff[1:-1])


@to_array
@nb.njit
def median(x):
	"""
	Returns the median of x
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""
	return np.median(x)


@to_array
@nb.njit
def mean(x):
	"""
	Returns the mean of x
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""
	return np.mean(x)


@to_array
@nb.njit
def length(x):
	"""
	Returns the length of x
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: int
	"""
	return len(x)


@to_array
@nb.njit
def standard_deviation(x):
	"""
	Returns the standard deviation of x
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""
	return np.std(x)


@to_array
@nb.njit
def variance(x):
	"""
	Returns the variance of x
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""
	return np.var(x)


@to_array
def skewness(x):
	"""
	Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
	moment coefficient G1).
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""
	return sp.stats.skew(x)


@to_array
def kurtosis(x):
	"""
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
	return sp.stats.kurtosis(x)


@to_array
@nb.njit
def maximum(x):
	"""
	Calculates the highest value of the time series x.
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""
	return np.max(x)


@to_array
@nb.njit
def minimum(x):
	"""
	Calculates the lowest value of the time series x.
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""
	return np.min(x)


@to_array
@nb.njit
def percentile(x, percent):
	return np.percentile(x, percent)


@to_array
def cid_ce(x, normalize=True):
	"""
	This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
	valleys etc.). It calculates the value of
	.. math::
		\\sqrt{ \\sum_{i=0}^{n-2lag} ( x_{i} - x_{i+1})^2 }
	.. rubric:: References
	|  [1] Batista, Gustavo EAPA, et al (2014).
	|  CID: an efficient complexity-invariant distance for time series.
	|  Data Mining and Knowledge Difscovery 28.3 (2014): 634-669.
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:param normalize: should the time series be z-transformed?
	:type normalize: bool
	:return: the value of this feature
	:return type: float
	"""

	if normalize:
		s = standard_deviation(x)
		if s != 0:
			x = (x - mean(x)) / s
		else:
			return 0.0

	x = np.diff(x)
	return np.sqrt(signal_energy(x))


@to_array
@nb.njit
def sample_entropy(x):
	"""
	Calculate and return sample entropy of x.
	.. rubric:: References
	|  [1] http://en.wikipedia.org/wiki/Sample_Entropy
	|  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:return: the value of this feature
	:return type: float
	"""

	sample_length = 1  # number of sequential points of the time series
	tolerance = 0.2 * np.std(x)  # 0.2 is a common value for r - why?

	n = len(x)
	prev = np.zeros(n)
	curr = np.zeros(n)
	A = np.zeros((1, 1))  # number of matches for m = [1,...,template_length - 1]
	B = np.zeros((1, 1))  # number of matches for m = [1,...,template_length]

	for i in range(n - 1):
		nj = n - i - 1
		ts1 = x[i]
		for jj in range(nj):
			j = jj + i + 1
			if abs(x[j] - ts1) < tolerance:  # distance between two vectors
				curr[jj] = prev[jj] + 1
				temp_ts_length = min(sample_length, curr[jj])
				for m in range(int(temp_ts_length)):
					A[m] += 1
					if j < n - 1:
						B[m] += 1
			else:
				curr[jj] = 0
		for j in range(nj):
			prev[j] = curr[j]

	N = n * (n - 1) / 2
	B = np.vstack((np.array([N]), B[0]))

	# sample entropy = -1 * (log (A/B))
	similarity_ratio = A / B
	se = -1 * np.log(similarity_ratio)
	se = np.reshape(se, -1)
	return se[0]


@to_array
def standard_error_mean(x):
	return sp.stats.sem(x)


@to_array
def c3(x, lag):
	"""
	This function calculates the value of
	.. math::
		\\frac{1}{n-2lag} \sum_{i=0}^{n-2lag} x_{i + 2 \cdot lag}^2 \cdot x_{i + lag} \cdot x_{i}
	which is
	.. math::
		\\mathbb{E}[L^2(X)^2 \cdot L(X) \cdot X]
	where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a measure of
	non linearity in the time series.
	.. rubric:: References
	|  [1] Schreiber, T. and Schmitz, A. (1997).
	|  Discrimination power of measures for nonlinearity in a time series
	|  PHYSICAL REVIEW E, VOLUME 55, NUMBER 5
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:param lag: the lag that should be used in the calculation of the feature
	:type lag: int
	:return: the value of this feature
	:return type: float
	"""
	from .helpers import _roll

	n = x.size
	if 2 * lag >= n:
		return 0
	else:
		return np.mean((_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0:(n - 2 * lag)])


@to_array
def augmented_dickey_fuller(x):
	"""
	The Augmented Dickey-Fuller test is a hypothesis test which checks whether a unit root is present in a time
	series sample. This feature calculator returns the value of the respective test statistic.
	See the statsmodels implementation for references and more details.
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:param param: contains dictionaries {"attr": x} with x str, either "teststat", "pvalue" or "usedlag"
	:type param: list
	:return: the value of this feature
	:return type: float
	"""
	from statsmodels.tsa.stattools import adfuller
	from numpy.linalg import LinAlgError
	from statsmodels.tools.sm_exceptions import MissingDataError

	try:
		res = adfuller(x)
	except (LinAlgError, ValueError, MissingDataError):
		res = np.NaN, np.NaN, np.NaN

	return {'teststat': res[0], 'pvalue': res[1], 'usedlag': res[2]}


@to_array
@nb.njit
def autocorrelation(x, lag):
	"""
	Calculates the autocorrelation of the specified lag, according to the formula [1]
	.. math::
		\\frac{1}{(n-l)\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)
	where :math:`n` is the length of the time series :math:`X_i`, :math:`\sigma^2` its variance and :math:`\mu` its
	mean. `l` denotes the lag.
	.. rubric:: References
	[1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:param lag: the lag
	:type lag: int
	:return: the value of this feature
	:return type: float
	"""
	# This is important: If a series is passed, the product below is calculated
	# based on the index, which corresponds to squaring the series.
	if len(x) < lag:
		return np.nan
	# Slice the relevant subseries based on the lag
	y1 = x[:(len(x) - lag)]
	y2 = x[lag:]
	# Subtract the mean of the whole series x
	x_mean = np.mean(x)
	# The result is sometimes referred to as "covariation"
	sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
	# Return the normalized unbiased covariance
	v = np.var(x)
	if -0.000000001 < v < 0.000000001:
		return np.NaN
	else:
		return sum_product / ((len(x) - lag) * v)


@to_array
@nb.njit
def large_standard_deviation(x, r):
	"""
	Boolean variable denoting if the standard dev of x is higher
	than 'r' times the range = difference between max and min of x.
	Hence it checks if
	.. math::
		std(x) > r * (max(X)-min(X))
	According to a rule of the thumb, the standard deviation should be a forth of the range of the values.
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:param r: the percentage of the range to compare with
	:type r: float
	:return: the value of this feature
	:return type: bool
	"""

	return np.std(x) > (r * (np.max(x) - np.min(x)))


@to_array
def number_cwt_peaks(x, n):
	"""
	This feature calculator searches for different peaks in x. To do so, x is smoothed by a ricker wavelet and for
	widths ranging from 1 to n. This feature calculator returns the number of peaks that occur at enough width scales
	and with sufficiently high Signal-to-Noise-Ratio (SNR)
	:param x: the time series to calculate the feature of
	:type x: numpy.ndarray
	:param n: maximum width to consider
	:type n: int
	:return: the value of this feature
	:return type: int
	"""

	return len(find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=ricker))


@to_array
def time_reversal_asymmetry_statistic(x, lag):
	"""
	This function calculates the value of
	.. math::
		\\frac{1}{n-2lag} \sum_{i=0}^{n-2lag} x_{i + 2 \cdot lag}^2 \cdot x_{i + lag} - x_{i + lag} \cdot  x_{i}^2
	which is
	.. math::
		\\mathbb{E}[L^2(X)^2 \cdot L(X) - L(X) \cdot X^2]
	where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
	promising feature to extract from time series.
	.. rubric:: References
	|  [1] Fulcher, B.D., Jones, N.S. (2014).
	|  Highly comparative feature-based time-series classification.
	|  Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.
	:param x: the time series to calculate the feature of
	:type x: pandas.Series
	:param lag: the lag that should be used in the calculation of the feature
	:type lag: int
	:return: the value of this feature
	:return type: float
	"""
	from .helpers import _roll
	n = len(x)
	if 2 * lag >= n:
		return 0

	one_lag = _roll(x, -lag)
	two_lag = _roll(x, 2 * -lag)
	return np.mean((two_lag * two_lag * one_lag - one_lag * x * x)[0:(n - 2 * lag)])


if __name__ == '__main__':
	import time
	import matplotlib.pyplot as plt

	x = np.linspace(0, 10000)
	y = x + np.random.normal(0, 100, x.shape)
	series = pd.Series(y)

	times = 2000000
	# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
	start = time.time()
	for n in range(times):
		ret = sample_entropy(series)
	end = time.time()
	print(f"Elapsed (with compilation) = {(end - start) / times}")
	print(ret)
