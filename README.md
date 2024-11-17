# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the California housing dataset and select the first 3 features (x) and the target along with another feature (y).

2.Split the data into training and testing sets (80% train, 20% test).

3.Scale the features (x) and targets (y) using StandardScaler.

4.Initialize a MultiOutputRegressor with an SGDRegressor and train it on the scaled training data.

5.Predict the targets on the test set and inverse scale the predictions and actual values.

6.Calculate and print the Mean Squared Error (MSE) between the predicted and actual values.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: S.PARTHASARATHI
RegisterNumber:  212223040144
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


data = fetch_california_housing()


x= data.data[:,:3]


y=np.column_stack((data.target,data.data[:,6]))

x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state =42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.fit_transform(y_test)

sgd = SGDRegressor(max_iter=1000, tol = 1e-3)

multi_output_sgd= MultiOutputRegressor(sgd)

multi_output_sgd.fit(x_train, y_train)

y_pred =multi_output_sgd.predict(x_test)

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
print(y_pred)
[[ 1.04860312 35.69231257]
 [ 1.49909521 35.72530255]
 [ 2.35760015 35.50646978]
 ...
 [ 4.47157887 35.06594388]
 [ 1.70991815 35.75406191]
 [ 1.79884624 35.34680017]]

mse = mean_squared_error(y_test,y_pred)

print("Mean Squared Error:",mse)

Mean Squared Error: 2.560165984862198

print("\nPredictions:\n",y_pred[:5])
Predictions:
 [[ 1.04860312 35.69231257]
 [ 1.49909521 35.72530255]
 [ 2.35760015 35.50646978]
 [ 2.73967825 35.37568192]
 [ 2.10914107 35.63894336]]
```

## Output:
MEAN :
![366126406-1e6c626f-8096-49d4-93b8-97ca749326d3](https://github.com/user-attachments/assets/1494fe8a-6fb6-49a7-8062-b8ba451a654d)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
