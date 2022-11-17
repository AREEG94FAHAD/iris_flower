# Testing the proposed model by using real data
import pandas as pd

# 1- define the new data
new_data = pd.DataFrame([{'sepal-length':5.3, 'sepal-width':3.7, 'petal-length':1.6, 'petal-width':0.22}])
new_data = new_data[['sepal-length','sepal-width','petal-length','petal-width']]

# 2- imort the scalar and knn model 

import joblib

model = joblib.load(r'Model/model.pickle')
scaler = joblib.load(r'Scaler/scaler.pickle')

# 3- normalize the data
new_data = scaler.transform(new_data)
predict_calass = model.predict(new_data)

print(predict_calass)

