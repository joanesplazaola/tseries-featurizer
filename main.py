import pandas as pd
from tseries import TimeSeriesFeaturizer
import numpy as np
import datetime

N = 3000

signal = np.sin(50 * np.pi * np.arange(N) / float(N / 2)) * 21

signal2 = np.sin(5 * np.pi * np.arange(N) / float(N / 2)) * 20
signal3 = np.sin(15 * np.pi * np.arange(N) / float(N / 2)) * 18 + 20
noise = np.random.normal(0, 1, N)
signal = signal + signal2 + signal3 + noise
base = datetime.datetime(2000, 1, 1)
time = np.array([base + datetime.timedelta(hours=i) for i in range(N)])

d = {
	'sensor1': signal,
	'sensor2': signal2 + noise,
	'sensor3': signal3 + noise,
	'sensor4': noise,
	'proba': list('s' * N),
	'time': time

}

data = pd.DataFrame(d)

tseries = TimeSeriesFeaturizer()

tranformed = tseries.featurize(data, time_column='time', n_jobs=16)
print(tranformed)
