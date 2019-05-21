from numpy.fft import fft
import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from scipy.signal import hilbert


def to_frequency(data, train_prev_data=None, test=False):
	"""
	Transforms time domain data to frequency data.

	When test is true, training set's length is taken into account in order to have the same indexes.

	:param train_prev_data: Dataframe with the training data after transforming.
	:param test: boolean that tells if we have to take the training set into account.
	:param data: pd.DataFrame or pd.Series in time-domain
	:return: pd.DataFrame in frequency domain
	"""
	if test:
		len = train_prev_data.shape[0] * 2
	else:
		len = data.shape[0]
	freq = pd.DataFrame()
	if isinstance(data, pd.Series):
		data = data.to_frame()
	for col in data:
		data[col].dropna(inplace=True)
		freq_data = abs(fft(data[col], len))
		half_size = freq_data.shape[0] / 2
		freq_data = freq_data[:int(half_size)]
		series = pd.Series(freq_data, name=col)
		freq = pd.concat([freq, series], axis=1)

	seconds_period = to_offset(data.index.inferred_freq).nanos / 10 ** 9

	max_index_freq = 1 / seconds_period
	freq_index = list(map(str, np.linspace(0, max_index_freq, freq.shape[0])))
	freq.index = freq_index
	return freq


def to_hilberts_freq(data, train_prev_data=None, test=False):
	hilbert_list = list()
	for col in data:
		trans_col = np.abs(hilbert(data[col].dropna()))
		na_col = pd.Series(name=col, data=np.append(trans_col, np.full(data.shape[0] - trans_col.shape[0], np.nan)))
		hilbert_list.append(na_col)
	hilbert_df = pd.concat(hilbert_list, axis=1)
	hilbert_df.index = data.index
	return to_frequency(hilbert_df, train_prev_data, test)


def wavelet_transform(data, wavelet):
	import pywt
	# Discrete Wavelet Transform
	ard, a = pywt.dwt(data, wavelet)
	# plt.plot(ard)

	return ard, a  # Returns approximation and detail coefficients


def inverse_wavelet_transform(appr_coeff, detail_coeff, wavelet):
	import pywt
	return pywt.idwt(appr_coeff, detail_coeff, wavelet)  # Returns approximation and detail coefficients
