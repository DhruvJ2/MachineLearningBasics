import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing , model_selection,  svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

#regression training and testing
x = np.array(df.drop(labels = ['label'], axis = 1))
y = np.array(df['label'])

x = preprocessing.scale(x)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2)
clf = LinearRegression()
clf.fit(x_train, y_train) #train in fit()
accuracy = clf.score(x_test, y_test) #test in score()

print(accuracy)