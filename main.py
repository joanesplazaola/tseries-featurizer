import pandas as pd
from ts_featurizer.base import TimeSeriesFeaturizer
import numpy as np
import datetime

# Hilbert analytical signal's fft


N = 3000

lista = []
for a in range(100):
	signal = np.sin(50 * np.pi * np.arange(N) / float(N / 2)) * 21

	signal2 = np.sin(5 * np.pi * np.arange(N) / float(N / 2)) * 20
	signal3 = np.sin(15 * np.pi * np.arange(N) / float(N / 2)) * 18 + 20
	noise = np.random.normal(0, 1, N)
	signal = signal + signal2 + signal3 + noise
	base = datetime.datetime(2000, 1, 1)
	time = np.array([base + datetime.timedelta(hours=i) for i in range(N)])
	noise2 = np.random.normal(0, 1, int(N / 2)),
	noise2 = np.append(noise2, np.full(int(N / 2), np.nan))
	d = {
		'sensor1': signal,
		'sensor2': signal2 + noise,
		'sensor3': signal3 + noise,
		'sensor4': noise,
		'proba': list('s' * N),
		'sensor5': noise2,
		'time': time

	}

	data = pd.DataFrame(d)
	lista.append(data)

tseries = TimeSeriesFeaturizer(check_na=False, collapse_columns=False)

transformed = tseries.featurize(lista[:5], time_column='time', n_jobs=1)
test_features = tseries.featurize(lista[5:], time_column='time', n_jobs=-1, apply_model=True)
