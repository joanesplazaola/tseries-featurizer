Sarrera
============

Zertarako erabili modulu hau?
------------------------------
tseries-featurizerren bidez denbora segiden ezaugarriak erauzi daitezke.
Now you want to calculate different characteristics such as the maximal or minimal temperature, the average temperature
or the number of temporary temperature peaks:

.. image:: ../images/introduction_ts_exa_features.png
   :scale: 70 %
   :alt: some characteristics of the time series
   :align: center

Without tsfresh, you would have to calculate all those characteristics by hand. With tsfresh this process is automated
and all those features can be calculated automatically.

Further tsfresh is compatible with pythons :mod:`pandas` and :mod:`scikit-learn` APIs, two important packages for Data
Science endeavours in python.

Zer egin daiteke ezaugarri hauekin?
------------------------------------

Erauzitako ezaugarriak denbora segiden deskribapen zehatzak direnez, sailkapen, erregresio, taldekatze... atazak egin
daitezke bertatik jasotako datuekin. Sarritan datuak edota prozesua hobeto ulertzeko balio du ezaugarri erauzketak,
hobetu nahi den prozesu baten aldagai bakoitzak zein garrantzi duen jakinda.

tseries-featurizer liburutegia honako proiektuetan erabili da:

    * Konpositeen sorrera prozesuaren egokitasunaren sailkapena

Zertarako ez du balio ts-featurizerrek?
---------------------------------------

Gaur egun, tseries-featurizerren bidez ezin da:

    * era banatuan exekutatu hainbat konputazio unitatetan barrena (natiboki), baina hau egin daiteke modeloa partekatuz gero, eskuz
    * modeloak entrenatu, hau modeloak eraiki aurreko pausua litzateke, era batera datuen aurre-prozesaketa, ondoren, `scikit-learn <http://scikit-learn.org/stable/>`_
      moduko liburutegien bidez modelo eraginkorrak sortu ahal izateko.


Zer beste liburutegi dago honetarako?
-------------------------------------

Matlaberako `hctsa <https://github.com/benfulcher/hctsa>`_ liburutegia aurki daiteke , eta hau pythonen erabiltzeko
`pyopy <https://github.com/strawlab/pyopy>`_ paketearen bidez egin daiteke. Horrez gain, pythonerako
`tsfresh <https://github.com/blue-yonder/tsfresh>`_ liburutegia erabili daiteke.
