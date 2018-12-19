from numpy.fft import fft
import pywt


def to_frequency(data):
	freq_data = abs(data.apply(fft))
	half_size = freq_data.shape[0] / 2
	freq_data = freq_data[:int(half_size)]
	return freq_data


def wavelet_transform(data, wavelet):
	# Discrete Wavelet Transform
	ard, a = pywt.dwt(data, wavelet)
	# plt.plot(ard)

	return ard, a  # Returns approximation and detail coefficients


def inverse_wavelet_transform(appr_coeff, detail_coeff, wavelet):
	return pywt.idwt(appr_coeff, detail_coeff, wavelet)  # Returns approximation and detail coefficients
