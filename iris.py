import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# read data file as csv and assigen the headernames

df = pd.read_csv('iris.data', names = headernames)
# print(df.head(3))

# select the dependent variable
x = df.iloc[:, :-1].values 

y = df.iloc[:,4].values

# normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# split the datframe into test and train 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.40)


from sklearn.neighbors import KNeighborsClassifier

k = 10
accuracy_result = []
for i in range(1,k+1):

    model = KNeighborsClassifier(n_neighbors = i).fit(X_train, y_train)
    yhat = model.predict(X_test)

    from sklearn import metrics
    # print("Train set Accuracy: ", metrics.accuracy_score(y_train, model.predict(X_train))*100)
    ac = metrics.accuracy_score(y_test, yhat)
    # print("Test set Accuracy: ",ac )

    accuracy_result.append(ac)

plt.bar(np.arange(1,k+1), [i*100 for i in accuracy_result])

print('The best accuracy achieved where k is equal to ', str(accuracy_result.index(max(accuracy_result))), "The accuracy is ",str(accuracy_result[accuracy_result.index(max(accuracy_result))]*100)+"%")
# plt.show()


# deploying

import joblib
import os
if not os.path.exists('Model'):
        os.mkdir('Model')
if not os.path.exists('Scaler'):
        os.mkdir('Scaler')
        
joblib.dump(model, r'Model/model.pickle')
joblib.dump(scaler, r'Scaler/scaler.pickle')

