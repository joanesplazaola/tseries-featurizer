# These are the names of the modules, and the functions of the modules we want to execute

default_conf = {

	'Time': {
		# Has to exist, must be string
		'class': 'tools.featurizers.TimeFeaturizer',

		# Optional, if exists has to be dict
		'previous_trans': {},

		# Features has to exist, must be dict
		# Each key must be a str(that can be converted to callable), and the item must be a list (of dicts)
		# Each dict of that list can only have three keys (max), named 'send_all_data', 'args' and 'kwargs'
		# send_all_data must be a boolean
		# args must be a list
		# kwargs must be a dictionary

		'features': {
			'numpy.mean': [],
			'numpy.median': [],
			'numpy.std': [],
			'numpy.min': [],
			'numpy.max': [],
			'scipy.stats.kurtosis': [],
			'scipy.stats.sem': [],
			'scipy.stats.skew': [],
			'numpy.percentile': [
				{'args': [50], },
				{'args': [95], 'kwargs': {}}
			],
			'tools.signal_energy': [],
		},
		# Optional, if exists, dict

	},

	'Frequency': {
		'class': 'tools.featurizers.FrequencyFeaturizer',

		'previous_trans': {
			'tools.to_frequency': {'args': [], 'kwargs': {}}
		},
		'features': {
			'tools.detect_peaks': [
				{'kwargs': {'show': False}, 'test': {
					'preprocessors': {
						'tools.item_list_from_tuple_list':
							{'item_index': 0, 'unique': True, 'sort': True, 'output_type': int}  # Keyword arguments
					},  # TODO OrderedDict??
				}}

			],
			'tools.signal_energy': [{}],

		},

	}
	# Each keyword of

	# 'Wavelet': {
	# 	'class': 'tools.featurizers.WaveletFeaturizer',
	# 	'previous_trans': {},
	# 	'features': {
	# 		'tools.wavelet_transform': [
	# 			{'args': [], 'kwargs': {'wavelet': 'db2'}}
	# 		]
	#
	# 	},
	# 	'apply_model_to_test': [],
	# },
	# 'AR': {
	# 	'class': 'tools.featurizers.AutoRegressionFeaturizer',
	# 	'previous_trans': {},
	# 	'features': {
	#
	# 		'tools.get_generic_AR': [
	# 			{'send_all_data': True, 'kwargs': {'max_coeffs': 3, }},
	#
	# 		],
	#
	# 	},
	# 	'apply_model_to_test': [],
	# },

}
