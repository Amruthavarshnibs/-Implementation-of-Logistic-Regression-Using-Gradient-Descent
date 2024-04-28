# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

2.Load Dataset: Load the dataset using pd.read_csv.

3.Remove irrelevant columns (sl_no, salary).

4.Convert categorical variables to numerical using cat.codes.

5.Separate features (X) and target variable (Y).

6.Define Sigmoid Function: Define the sigmoid function.

7.Define Loss Function: Define the loss function for logistic regression.

8.Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

9.Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

10.Define Prediction Function: Implement a function to predict the output based on the learned parameters.

11.Evaluate Accuracy: Calculate the accuracy of the model on the training data.

12.Predict placement status for a new student with given feature values (xnew).

13.Print Results: Print the predictions and the actual values (Y) for comparison.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: amrutha varshni B S
RegisterNumber: 2122220007

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:/Users/admin/Desktop/Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop("salary",axis=1)
dataset ["gender"] = dataset ["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset ["hsc_b"].astype('category')
dataset ["degree_t"] = dataset ["degree_t"].astype('category')
dataset ["workex"] = dataset ["workex"].astype('category')
dataset["specialisation"] = dataset ["specialisation"].astype('category')
dataset ["status"] = dataset["status"].astype('category')
dataset ["hsc_s"] = dataset ["hsc_s"].astype('category')
dataset.dtypes
dataset ["gender"] = dataset ["gender"].cat.codes
dataset ["ssc_b"] = dataset["ssc_b"].cat.codes
dataset ["hsc_b"] = dataset ["hsc_b"].cat.codes
dataset ["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset ["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
```


## Output:

### Dataset

![EXP 5 dataset](https://github.com/23003250/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331462/9b076906-d52f-4c8d-83d9-0155b55e49b2)

### dataset.dtypes

![exp 5 dtypes](https://github.com/23003250/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331462/e5508a88-6137-407e-8d44-4fe46ad5bff6)

### dataset

![dataset 2](https://github.com/23003250/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331462/aa88f9aa-acc4-4cc7-aa7b-196b89462ace)


### Y

![exp 5 Y](https://github.com/23003250/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331462/6ddf9b61-cb4c-4d68-b84d-35e587f7a571)

#### y_pred

![exp 5 y pred](https://github.com/23003250/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331462/aca1438c-fa44-418a-beca-e5285df943df)

#### Y

![exp 5 y 2](https://github.com/23003250/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331462/d4678ccd-2c8b-4fed-9b23-277abbe24112)

#### y_prednew

![exp 5 y pred 2](https://github.com/23003250/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331462/af74d211-62ee-48ea-ba62-4723adb6853b)

#### y_prednew

![exp 5 y pred 2](https://github.com/23003250/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331462/324b4488-61f9-4bd2-b99a-a493204563d5)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

