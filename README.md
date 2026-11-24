# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the dataset containing house features, house price, and number of occupants.
2. Split the dataset into training and testing sets, then standardize the input features.
3. Train the SGD Regressor model using the training data and predict the house price and number of occupants.
4. Display the predicted values and evaluate the model performance.

## Program:
```
import numpy as np

from sklearn.linear_model import SGDRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# Input Features
# Area (sq.ft), Bedrooms, Age (years)
X = np.array([
    [1000, 2, 5],
    [1200, 3, 4],
    [1500, 3, 3],
    [1800, 4, 2],
    [2000, 4, 1],
    [2200, 5, 2],
    [2500, 5, 1],
    [2700, 6, 1]
])

# Output
# Price (Lakhs), Number of Occupants
Y = np.array([
    [40, 3],
    [48, 4],
    [60, 4],
    [72, 5],
    [80, 5],
    [90, 6],
    [100, 6],
    [110, 7]
])

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# SGD Regressor
model = MultiOutputRegressor(
    SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
)

# Train
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

print("Actual Values:")

print(Y_test)

print("\nPredicted Values:")

print(np.round(Y_pred, 2))

# Predict for a new house
new_house = np.array([[1600, 3, 2]])

new_house = scaler.transform(new_house)

prediction = model.predict(new_house)

print("\nPrediction for New House")

print("Predicted Price (Lakhs):", round(prediction[0][0], 2))

print("Predicted Occupants:", round(prediction[0][1]))
```

## Outpmport numpy as np
<img width="262" height="193" alt="image" src="https://github.com/user-attachments/assets/ae387ca1-e40e-4963-abef-2428b4750a21" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
