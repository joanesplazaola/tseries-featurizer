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
	arima_mod_params = None
	try:
		arima_mod = ARIMA(data, order).fit(disp=0)
		aic = arima_mod.aic
		arima_mod_params = arima_mod.params
	except Exception as e:
		pass
	return aic, arima_mod_params


def get_generic_AR(data, max_coeffs, all_data):
	"""

	:param data:
	:param max_coeffs:
	:param all_data:
	:return:
	"""
	# model, orders = get_model_and_possible_orders(model, max_coeffs, )
	param_list = list()

	rest_data = all_data.drop([data.name], axis='columns')
	best_score, best_cfg, best_params = float("inf"), None, None
	orders = get_possible_orders([max_coeffs, max_coeffs], max_coeffs)
	orders = np.concatenate((np.insert(orders, 1, 1, axis=1), np.insert(orders, 1, 0, axis=1)), )

	for order in orders:
		aic, params = get_orders_aic(order, data)
		if aic < best_score:
			best_score, best_cfg, best_params = aic, order, params

	param_list.append(best_params.values)

	for column in rest_data:
		aic, best_params = get_orders_aic(best_cfg, rest_data[column])
		param_list.append(best_params.values)

	df_params = pd.DataFrame(param_list)

	std_sum = df_params.std().sum()

	return best_cfg, std_sum
