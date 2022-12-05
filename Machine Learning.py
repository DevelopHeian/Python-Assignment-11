


import pandas as pd
df = pd.read_csv('housepricedata.csv')
df


     
LotArea	OverallQual	OverallCond	TotalBsmtSF	FullBath	HalfBath	BedroomAbvGr	TotRmsAbvGrd	Fireplaces	GarageArea	AboveMedianPrice
0	8450	7	5	856	2	1	3	8	0	548	1
1	9600	6	8	1262	2	0	3	6	1	460	1
2	11250	7	5	920	2	1	3	6	1	608	1
3	9550	7	5	756	1	0	3	7	1	642	0
4	14260	8	5	1145	2	1	4	9	1	836	1
5	14115	5	5	796	1	1	1	5	0	480	0
6	10084	8	5	1686	2	0	3	7	1	636	1
7	10382	7	6	1107	2	1	3	7	2	484	1
8	6120	7	5	952	2	0	2	8	2	468	0
9	7420	5	6	991	1	0	2	5	2	205	0
10	11200	5	5	1040	1	0	3	5	0	384	0
11	11924	9	5	1175	3	0	4	11	2	736	1
12	12968	5	6	912	1	0	2	4	0	352	0
13	10652	7	5	1494	2	0	3	7	1	840	1
14	10920	6	5	1253	1	1	2	5	1	352	0
15	6120	7	8	832	1	0	2	5	0	576	0
16	11241	6	7	1004	1	0	2	5	1	480	0
17	10791	4	5	0	2	0	2	6	0	516	0
18	13695	5	5	1114	1	1	3	6	0	576	0
19	7560	5	6	1029	1	0	3	6	0	294	0
20	14215	8	5	1158	3	1	4	9	1	853	1
21	7449	7	7	637	1	0	3	6	1	280	0
22	9742	8	5	1777	2	0	3	7	1	534	1
23	4224	5	7	1040	1	0	3	6	1	572	0
24	8246	5	8	1060	1	0	3	6	1	270	0
25	14230	8	5	1566	2	0	3	7	1	890	1
26	7200	5	7	900	1	0	3	5	0	576	0
27	11478	8	5	1704	2	0	3	7	1	772	1
28	16321	5	6	1484	1	0	2	6	2	319	1
29	6324	4	6	520	1	0	1	4	0	240	0
...	...	...	...	...	...	...	...	...	...	...	...
1430	21930	5	5	732	2	1	4	7	1	372	1
1431	4928	6	6	958	2	0	2	5	0	440	0
1432	10800	4	6	656	2	0	4	5	0	216	0
1433	10261	6	5	936	2	1	3	8	1	451	1
1434	17400	5	5	1126	2	0	3	5	1	484	0
1435	8400	6	9	1319	1	1	3	7	1	462	1
1436	9000	4	6	864	1	0	3	5	0	528	0
1437	12444	8	5	1932	2	0	2	7	1	774	1
1438	7407	6	7	912	1	0	2	6	0	923	0
1439	11584	7	6	539	2	1	3	6	1	550	1
1440	11526	6	7	588	2	0	3	11	1	672	1
1441	4426	6	5	848	1	0	1	3	1	420	0
1442	11003	10	5	1017	2	1	3	10	1	812	1
1443	8854	6	6	952	1	0	2	4	1	192	0
1444	8500	7	5	1422	2	0	3	7	0	626	1
1445	8400	6	5	814	1	0	3	6	0	240	0
1446	26142	5	7	1188	1	0	3	6	0	312	0
1447	10000	8	5	1220	2	1	3	8	1	556	1
1448	11767	4	7	560	1	1	2	6	0	384	0
1449	1533	5	7	630	1	0	1	3	0	0	0
1450	9000	5	5	896	2	2	4	8	0	0	0
1451	9262	8	5	1573	2	0	3	7	1	840	1
1452	3675	5	5	547	1	0	2	5	0	525	0
1453	17217	5	5	1140	1	0	3	6	0	0	0
1454	7500	7	5	1221	2	0	2	6	0	400	1
1455	7917	6	5	953	2	1	3	7	1	460	1
1456	13175	6	6	1542	2	0	3	7	2	500	1
1457	9042	7	9	1152	2	0	4	9	2	252	1
1458	9717	5	6	1078	1	0	2	5	0	240	0
1459	9937	5	6	1256	1	1	3	6	0	276	0
1460 rows × 11 columns

The dataset that we have now is in what we call a pandas dataframe. To convert it to an array, simply access its values:


dataset = df.values


     

dataset


     
array([[ 8450,     7,     5, ...,     0,   548,     1],
       [ 9600,     6,     8, ...,     1,   460,     1],
       [11250,     7,     5, ...,     1,   608,     1],
       ...,
       [ 9042,     7,     9, ...,     2,   252,     1],
       [ 9717,     5,     6, ...,     0,   240,     0],
       [ 9937,     5,     6, ...,     0,   276,     0]], dtype=int64)
Now, we split the dataset into our input features and the label we wish to predict.


X = dataset[:,0:10]
Y = dataset[:,10]


     
Normalizing our data is very important, as we want the input features to be on the same order of magnitude to make our training easier. We'll use a min-max scaler from scikit-learn which scales our data to be between 0 and 1.


from sklearn import preprocessing


     

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


     
D:\Anaconda3\envs\intuitive-deep-learning\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by MinMaxScaler.
  warnings.warn(msg, DataConversionWarning)

X_scale


     
array([[0.0334198 , 0.66666667, 0.5       , ..., 0.5       , 0.        ,
        0.3864598 ],
       [0.03879502, 0.55555556, 0.875     , ..., 0.33333333, 0.33333333,
        0.32440056],
       [0.04650728, 0.66666667, 0.5       , ..., 0.33333333, 0.33333333,
        0.42877292],
       ...,
       [0.03618687, 0.66666667, 1.        , ..., 0.58333333, 0.66666667,
        0.17771509],
       [0.03934189, 0.44444444, 0.625     , ..., 0.25      , 0.        ,
        0.16925247],
       [0.04037019, 0.44444444, 0.625     , ..., 0.33333333, 0.        ,
        0.19464034]])
Lastly, we wish to set aside some parts of our dataset for a validation set and a test set. We use the function train_test_split from scikit-learn to do that.


from sklearn.model_selection import train_test_split


     

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)


     

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)


     

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
