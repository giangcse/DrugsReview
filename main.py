import pandas as pd
dt_Train = pd.read_csv('drugsComTestProcessed.csv')
dt_Test = pd.read_csv('drugsComTrainProcessed.csv')
dt_Test[1:5]
dt_Train[1:5]
x_train = dt_Train[['Id', 'vaderReviewScore']]
y_train = dt_Train[['ratingSentimentLabel']]

x_test = dt_Test[['Id', 'vaderReviewScore']]
y_test = dt_Test[['ratingSentimentLabel']]

y_train[1:5]
y_test[1:5]
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(random_state=0)
clf_gini.fit(x_train, y_train)

y_pred = clf_gini.predict(x_test)

from sklearn.metrics import accuracy_score
print "Accuracy score: %3.2f" %(accuracy_score(y_test, y_pred)*100)
