# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S Ashwinkumar
RegisterNumber:212222040020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)


#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:


![p1](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120731469/30494b4f-58e9-492a-8a46-d889b40a7929)


![p2](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120731469/63d356bd-4387-4191-9c30-97e02e6ab95d)



![p3](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120731469/b12d1ae9-a3dd-49f1-a141-c9fe865b5f02)



![p4](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120731469/abbcb2df-f9a6-4df0-8b0e-c2d21885bdd0)



![p5](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120731469/aa12a2ee-2894-4809-a533-010a62f800e0)







![p6](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120731469/c1b2286a-38c5-4728-8bba-a3d4ec5d758a)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
