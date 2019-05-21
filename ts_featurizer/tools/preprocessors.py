import pandas as pd
from .helpers import get_attr_from_module
import itertools


def item_list_from_tuple_list(data, item_index=0, unique=False, sort=True, output_type=int):
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
	:param evaluated_field:
	:param evaluator_fn:
	:param model_field:
	:return:
	"""
	evaluator_fn = get_attr_from_module(evaluator_fn)

	evaluated_col = data.filter(regex=f'{evaluated_field}$')
	best_index = evaluator_fn(evaluated_col.iloc[:, 0].tolist())  # This function must always return an index
	selected_row = data.iloc[best_index]
	selected_frame = selected_row.to_frame().T
	return selected_frame.loc[:, selected_frame.columns.str.endswith(model_field)].iloc[0, 0]


def get_one_from_cols(data, cols):
	"""
	Takes a column from many.
	:param data:
	:param cols:
	:return:
	"""
	val_list = list()
	for col in cols:
		evaluated_col = data.filter(regex=f'{col}$')
		val_list.append(evaluated_col.iloc[0, 0])
	return val_list


def get_list_from_columns(data):
	"""
	Takes the text after the underscore in each of the data columns and inserts them to a list.
	:param data: DataFrame with columns.
	:return: list of column parameters in str.
	"""
	columns = data.columns.tolist()
	data_list = list(map(lambda x: str(x.split('_')[-1]), columns))
	return data_list


def get_most_important_freqs(data):
	"""
	Selects the most important frequencies, those that have been detected as peak in any of the time series.
	:param data:
	:return:
	"""
	from .features import detect_peaks

	freqs = [list(detect_peaks(data[col]).keys()) for col in data]

	important_freqs = set(itertools.chain(*freqs))

	return important_freqs


def get_AR_model(data, max_coeffs):
	"""
	Gets the AR model of one of the columns randomly,
	taking into account the columns contain more or less similar data.
	:param data:
	:param max_coeffs:
	:return:
	"""
	from random import shuffle
	from .helpers import ARUtils
	col_list = list(data.columns)
	shuffle(col_list)
	for col in col_list:
		fit_model = ARUtils.get_best_order(data[col].dropna(), max_coeffs)
		if fit_model is not None:
			break
	return fit_model
