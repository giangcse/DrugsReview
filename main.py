import pandas as pd 
from sklearn.model_selection import KFold
X_train = pd.read_csv('drugsComTrain_raw.csv', usecols=['drugName', 'condition', 'usefulCount'])
X_test = pd.read_csv('drugsComTest_raw.csv', usecols=['drugName', 'condition', 'usefulCount'])

Y_train = pd.read_csv('drugsComTrain_raw.csv', usecols=['rating'])
Y_test = pd.read_csv('drugsComTest_raw.csv', usecols=['rating'])

from sklearn.tree import DecisionTreeClassifier
duy = DecisionTreeClassifier()
duy.fit(X_train, Y_train)

Y_pred = duy.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, Y_pred)*100
