# These are the names of the modules, and the functions of the modules we want to execute

default_conf = {

	'train': {
		'Frequency': {
			'class': 'tools.featurizers.FrequencyFeaturizer',
			'previous_trans': {
				'tools.to_frequency': {'args': [], 'kwargs': {}}
			},
			'features': {
				'tools.detect_peaks': [
					{'kwargs': {'show': True}}

				],
				'tools.signal_energy': [{}],

			},
		},

		'Wavelet': {
			'class': 'tools.featurizers.WaveletFeaturizer',
			'previous_trans': {},
			'features': {
				'tools.wavelet_transform': [
					{'args': [], 'kwargs': {'wavelet': 'db2'}}
				]

			},
		},
		'AR': {
			'class': 'tools.featurizers.AutoRegressionFeaturizer',
			'previous_trans': {},
			'features': {

				'tools.get_generic_AR': [
					{'send_all_data': True, 'kwargs': {'max_coeffs': 3, }},

				],

			},
		},

	},
	'test': {
		'Time': {
			'class': 'tools.featurizers.TimeFeaturizer',
			'previous_trans': {},
			'features': {
				'numpy.mean': [{}],
				'numpy.median': [{}],
				'numpy.std': [{}],
				'numpy.min': [{}],
				'numpy.max': [{}],
				'scipy.stats.kurtosis': [{}],
				'scipy.stats.sem': [{}],
				'scipy.stats.skew': [{}],
				'numpy.percentile': [
					{'args': [50], },  # These are each one of the calls to the function with different args
					{'args': [95], 'kwargs': {}}
				],
				'tools.signal_energy': [{}],
			},

		},
		'Frequency': {
			'class': 'tools.featurizers.FrequencyFeaturizer',
			'previous_trans': {
				'tools.to_frequency': {'args': [], 'kwargs': {}}
			},
			'features': {
				'tools.detect_peaks': [
					{'kwargs': {'show': True}}

				],
				'tools.signal_energy': [{}],

			},
		},

	}
}
