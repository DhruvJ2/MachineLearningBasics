import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import pickle

#Preproccess + EDA + Feature Selection
df = pd.read_csv('placement.csv')
df = df.iloc[:,1:]

plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
# plt.show()
##locical Regression is used

#Extracting input and ouput columns
X=df.iloc[:,0:2]
y=df.iloc[:,-1]

#Train test split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1)

# print(y_train,y_test)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

clf=LogisticRegression()
#model training
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
# clf.predict(y_test)
print(accuracy_score(y_pred,y_test))
plot_decision_regions(X_train, y_train.values,clf=clf, legend=2)
# plt.show()

#deploy the file created by pickle
pickle.dump(clf,open('model.pkl','wb'))