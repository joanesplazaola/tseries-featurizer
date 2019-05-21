# These are the names of the modules, and the functions of the modules we want to execute

from .config_factory import get_errors


class DefaultConfiguration:

	def __init__(self):
		self.feature_paths = ['ts_featurizer.tools', ]

		self.configuration = {

			'Time': {
				# Has to exist, must be string
				'class': 'ts_featurizer.tools.featurizers.TimeFeaturizer',

				# Optional, if exists has to be dict
				'previous_trans': {},

				# Features has to exist, must be dict
				# Each key must be a str(that can be converted to callable), and the item must be a list (of dicts)
				# Each dict of that list can only have three keys (max), named 'send_all_data', 'args' and 'kwargs'
				# send_all_data must be a boolean
				# args must be a list
				# kwargs must be a dictionary

				'features': {

					'standard_error_mean': [],
					'signal_energy': [],
					'maximum': [],
					'minimum': [],
					'skewness': [],
					'kurtosis': [],
					'standard_deviation': [],
					'variance': [],
					'length': [],
					'mean': [],
					'median': [],
					'mean_second_derivative_central': [],
					'cid_ce': [],
					'sample_entropy': [],
					'augmented_dickey_fuller': [],
					'c3': [{'args': [r]} for r in range(1, 4)],
					'large_standard_deviation': [{"args": [r * 0.05]} for r in range(1, 20)],
					'percentile': [{'args': [percent]} for percent in [50, 75, 95, 99]],
					"time_reversal_asymmetry_statistic": [{"args": [lag]} for lag in range(1, 4)],
					# 'number_cwt_peaks': [{'args': [r]} for r in [1, 3, 5, 10, 50]],

					# 'autocorrelation':  [{'args': [r]} for r in range(1, 4)],
					# 'first_location_of_minimum': [],
				},
				# Optional, if exists, dict

			},

			'Frequency': {
				'class': 'ts_featurizer.tools.featurizers.FrequencyFeaturizer',

				'previous_trans': {
					'ts_featurizer.tools.to_frequency': {'args': [], 'kwargs': {}}
				},
				'features': {
					'detect_peaks': [
						{
							'test': {
								'preprocessors': {
									'ts_featurizer.tools.get_list_from_columns': {}  # Keyword arguments
								},  # TODO OrderedDict??
							},
							'model': {
								'preprocessor': 'ts_featurizer.tools.get_most_important_freqs',
								'kwargs': {}
							}
						}, ],

					'standard_error_mean': [],
					'signal_energy': [],
					'maximum': [],
					'minimum': [],
					'skewness': [],
					'kurtosis': [],
					'standard_deviation': [],
					'variance': [],
					'length': [],
					'mean': [],
					'median': [],

					# 'number_cwt_peaks': [{'args': [r]} for r in [1, 3, 5, 10, 50]],

				},

			},

			'Hilbert': {
				'class': 'ts_featurizer.tools.featurizers.HilbertFeaturizer',

				'previous_trans': {
					'ts_featurizer.tools.to_hilberts_freq': {'args': [], 'kwargs': {}}
				},
				'features': {
					'detect_peaks': [
						{
							'test': {
								'preprocessors': {
									'ts_featurizer.tools.get_list_from_columns': {}  # Keyword arguments
								},  # TODO OrderedDict??
							},
							'model': {
								'preprocessor': 'ts_featurizer.tools.get_most_important_freqs',
								'kwargs': {}
							}
						}, ],

					'standard_error_mean': [],
					'signal_energy': [],
					'maximum': [],
					'minimum': [],
					'skewness': [],
					'kurtosis': [],
					'standard_deviation': [],
					'variance': [],
					'length': [],
					'mean': [],
					'median': [],
					# 'number_cwt_peaks': [{'args': [r]} for r in [1, 3, 5, 10, 50]],

				},

			},
			# 'AR': {
			# 	'class': 'ts_featurizer.tools.featurizers.AutoRegressionFeaturizer',
			# 	'previous_trans': {},
			# 	'features': {
			#
			# 		'get_AR_params': [
			# 			{
			# 				'test': {
			# 					'preprocessors': {
			# 						'ts_featurizer.tools.get_one_from_cols': {
			# 							'cols': ['best_order_p', 'best_order_d', 'best_order_q']}
			# 					},
			# 				},
			# 				'model': {
			# 					'preprocessor': 'ts_featurizer.tools.get_AR_model',
			# 					'kwargs': {'max_coeffs': 3, }
			# 				}
			#
			# 			}, ],
			#
			# 	},
			# },

			# 'Wavelet': {
			# 	'class': 'ts_featurizer.tools.featurizers.WaveletFeaturizer',
			# 	'previous_trans': {},
			# 	'features': {
			# 		'ts_featurizer.tools.wavelet_transform': [
			# 			{'args': [], 'kwargs': {'wavelet': 'db2'}}
			# 		]
			#
			# 	},
			# },

		}
		self.errors = get_errors(self)

		if len(self.errors) > 0:
			raise ValueError(self.errors)

	def add_featurizer(self):
		pass
