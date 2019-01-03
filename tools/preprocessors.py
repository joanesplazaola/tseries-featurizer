import pandas as pd


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
