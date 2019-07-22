import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


data_path = "heart.csv" 
data = pd.read_csv(data_path, index_col=0)
print(list(data.columns))


feature_names = ['sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                 'slope', 'ca', 'thal', 'target']

'''
feature_names =['sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 
                'slope', 'ca', 'thal', 'target']

'''
X = data[feature_names]
y = data.oldpeak
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

print(' \nmean squared error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

