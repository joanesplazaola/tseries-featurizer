from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from tools import get_possible_orders
import pandas as pd


def get_orders_aic(order, data):
	"""

	:param order:
	:param data:
	:return:
	"""
	aic = float('inf')
	p, q, c = None, None, None
	try:
		arima_mod = ARIMA(data, order).fit(disp=0)
		aic = arima_mod.aic
		p, q, c = arima_mod.arparams, arima_mod.maparams, arima_mod.params.const
	except Exception as e:
		pass
	return aic, (p, q, [c])


def get_best_orders(data, max_coeffs):
	"""

	:param data:
	:param max_coeffs:
	:return:
	"""
	best_score, best_cfg, best_params = float("inf"), None, None
	orders = get_possible_orders([max_coeffs, max_coeffs], max_coeffs)
	orders = np.concatenate((np.insert(orders, 1, 1, axis=1), np.insert(orders, 1, 0, axis=1)), )

	for order in orders:
		aic, params = get_orders_aic(order, data)
		if aic < best_score:
			best_score, best_cfg, best_params = aic, order, params
	return best_cfg, best_params


def get_generic_AR(data, max_coeffs, all_data, fit_model=None, test=False):
	"""

	:param data:
	:param max_coeffs:
	:param all_data:
	:param fit_model:
	:param test:
	:return:
	"""
	param_list = list()
	params = 0, 0, 0
	if not test:
		fit_model, params = get_best_orders(data, max_coeffs)

	for column in all_data:
		_, best_params = get_orders_aic(fit_model, all_data[column])
		if best_params[0] is None:
			continue
		flat_list = [item for sublist in best_params for item in sublist]
		param_list.append(flat_list)

	df_params = pd.DataFrame(param_list)

	std_sum = df_params.std().sum()

	ret = {'best_order': fit_model, 'score': std_sum, 'constant': params[2][0]}
	to_dict = lambda params, title: {f'{title}_{index}': param for index, param in enumerate(params)}
	ar_params = to_dict(params[0], 'ar_params')
	ma_params = to_dict(params[1], 'ma_params')
	ret.update({**ar_params, **ma_params})

	return ret
