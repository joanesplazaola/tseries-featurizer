.. _data-formats-label:

Datu formatua
==============

tseries-featurizerreko ``featurize`` funtzioak sarrera moduan espero dituen datuen formatua azalduko da jarraian.

Irteerako datuek berriz, bi formatu desberdin izan dezakete, erabiltzaileak erabilitako parametroen arabera.



Bai sarrerako, zein irteerako objektuak :class:`pandas.DataFrame` motakoak izango dira. Sarrerako datuen kasuan,
:class:`pandas.DataFrame` zerrenda bat izango da, elementu bakoitza denbora segida bat izanik, irteerako kasuan, berriz,
:class:`pandas.DataFrame` bakarra izango da, erauzitako ezaugarri bakoitzarekin


Garrantzitsua: Sarrerako datuek ezingo dute ``NaN``, ``Inf`` edota ``-Inf`` baliorik izan.


Sarrerako formatua: DataFrame zerrenda
---------------------------------------


Adibidez: Imaginatu hainbat etxetako tenperatura eta hezetasuna neurtzen duten sentsoreen datuak dauzkagula, eta
 neurketa hauek denboran zehar izaniko bilakaera deskribatzen dutela. Ezaugarriak tseries-featurizer liburutegia erabiliz
 erauzteko, etxe bakoitzeko denbora segidak ``DataFrame`` batean jarri beharko dira, eta denak pythoneko zerrenda batean
 bildu. Jarraian ikus daitezke zerrendako bi elementu:

    +----+---------+-------------+------------+
    | id |   time  | tenperatura | hezetasuna |
    +----+---------+-------------+------------+
    |  A |    t1   |  tenp(A,t1) |  hez(A,t1) |
    +----+---------+-------------+------------+
    |  A |    t2   |  tenp(A,t2) |  hez(A,t2) |
    +----+---------+-------------+------------+
    |  A |    t3   |  tenp(A,t3) |  hez(A,t3) |
    +----+---------+-------------+------------+
    |  A |    t4   |  tenp(A,t4) |  hez(A,t4) |
    +----+---------+-------------+------------+

    +----+---------+-------------+------------+
    | id |   time  | tenperatura | hezetasuna |
    +----+---------+-------------+------------+
    |  B |    t1   |  tenp(B,t1) |  hez(B,t1) |
    +----+---------+-------------+------------+
    |  B |    t2   |  tenp(B,t2) |  hez(B,t2) |
    +----+---------+-------------+------------+
    |  B |    t3   |  tenp(B,t3) |  hez(B,t3) |
    +----+---------+-------------+------------+

liburutegiak `time` izena duen aldagaia identifikatuko du denbora aldagai moduan, eta hau egongo ez balitz berak jarriko
lioke lagintze-denbora jakin bat. Horrez gain, aldakorrak ez diren aldagaiak (identifikatzailea kasu honetan) ez dira
ezaugarri erauzketarako kontuan izango.

Irteerako formatuak
--------------------

Irteerako datuen formatua, esan bezala, :class:`pandas.DataFrame` motakoa izango da, baina instantzia sortzean
`collapse_columns` parametroaren balioaren arabera, bi formatu izango ditu.

Collapsed columns
------------------

+--------------+--------------+-----+-------------------+-------------------+-------------+-------------+-----+------------------+------------------+
| tenp_Time_f1 | tenp_Time_f2 | ... | tenp_Frequency_f1 | tenp_Frequency_f2 | hez_Time_f1 | hez_Time_f2 | ... | hez_Frequency_f1 | hez_Frequency_f1 |
+--------------+--------------+-----+-------------------+-------------------+-------------+-------------+-----+------------------+------------------+
|      ...     |      ...     | ... |        ...        |        ...        |     ...     |     ...     | ... |        ...       |        ...       |
+--------------+--------------+-----+-------------------+-------------------+-------------+-------------+-----+------------------+------------------+
|      ...     |      ...     | ... |        ...        |        ...        |     ...     |     ...     | ... |        ...       |        ...       |
+--------------+--------------+-----+-------------------+-------------------+-------------+-------------+-----+------------------+------------------+

Not collapsed columns
----------------------

+-----------------------------------------------------+-----------------------------------------------------+
|                     tenperatura                     |                      hezetasuna                     |
+-----------------------------------------------------+-----------------------------------------------------+
|          Time         | aaa |       Frequency       |          Time         | aaa |       Frequency       |
+-----------------------+-----+-----------------------+-----------------------+-----+-----------------------+
| feature_1 | feature_2 | ... | feature_1 | feature_2 | feature_1 | feature_2 | ... | feature_1 | feature_2 |
+-----------+-----------+-----+-----------+-----------+-----------+-----------+-----+-----------+-----------+
|    ...    |    ...    | ... |    ...    |    ...    |    ...    |    ...    | ... |    ...    |    ...    |
+-----------+-----------+-----+-----------+-----------+-----------+-----------+-----+-----------+-----------+
|    ...    |    ...    | ... |    ...    |    ...    |    ...    |    ...    | ... |    ...    |    ...    |
+-----------+-----------+-----+-----------+-----------+-----------+-----------+-----+-----------+-----------+
