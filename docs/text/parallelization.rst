.. _parallelization-label:

Paralelizazioa
===============

Tseries-featurizerrekin, ezaugarri erauzketa paraleloki egin daiteke. Horretarako, `n_jobs` parametroa erabiliko da.


Prozesu banaketa
-------------------------------------
Momentu honetan, datuak modelatu osteko instantzia gordez gero, `pickle` erabiliz, adibidez, gai gara datuak
ordenagailu desberdinetan momentu berean prozesatzeko. Hala ere, ordenagailu hauetan gordetako instantzia eskuz sartu beharko
litzateke, eta prozesua martxan jarri nahi den fitxategiekin. Ordenagailu batean, edota 200 ordenagailutan lortuko genukeen
emaitza bera litzateke, hasieran sortutako modeloa dela medio.

Etorkizunera begira, baina, klusterretan natiboki funtzionatzeko modua gehitzea oso interesgarria litzateke, hau da, hasieran
modelatzea soilik ordenagailu batean egin ostean, modelo hori klusterreko nodoetara pasa, eta nodo bakoitzak fitxategi
jakin batzuetan aplikatuz lan banaketa automatikoa egitea.
