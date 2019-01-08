import pandas as pd
import numpy as np


# TODO This has to be generalized or many functions need to be done
def item_list_from_tuple_list(data, item_index=0, unique=False, sort=True, output_type=int):
	"""
	Takes a pd.Series of tuples(frequency, amplitude) and returns a pd.Series of the frequencies
	:param signals_peaks: Series of tuples(frequency, amplitude)
	:type signals_peaks: pandas.Series
	:return: Series of frequencies
	"""
	item_list = data.apply(lambda x: list(list(zip(*x))[item_index])).apply(pd.Series).stack()
	if unique:
		item_list = item_list.unique()

	item_list = list(map(output_type, item_list))
	if sort:
		item_list = sorted(item_list)
	return item_list


def get_evaluated_function(data, evaluated_field, evaluator_fn, model_field):
	"""

	:param data:
	:return:
	"""
	# TODO This has to be tested and set correctly

	selected_row = data.iloc[np.argmin(data.loc[:, data.columns.str.endswith(evaluated_field)].T.squeeze().tolist())]

	return selected_row.to_frame().T.loc[:, selected_row.to_frame().T.columns.str.endswith(model_field)].iloc[0, 0]


def get_list_from_columns(data):
	columns = data.columns.tolist()
	data_list = list(map(lambda x: int(x.split('_')[-1]), columns))
	return data_list
