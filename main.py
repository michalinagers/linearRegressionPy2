import pandas as pd  #For handling dataframes
import matplotlib.pyplot as plt  #For plotting graphs
import seaborn as sns  #For enhanced visualizations
from sklearn.model_selection import train_test_split  #Splitting data into training/testing sets
from sklearn.linear_model import LinearRegression  #Linear regression model
from sklearn.metrics import mean_squared_error, r2_score  #Model evaluation metrics
import numpy as np  #For numerical operations

#Load dataset
House = pd.read_csv('/content/drive/MyDrive/USA_Housing.csv')

#Display basic info
print(House.head())
print(House.info())

#Selecting relevant features
housing = House[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
                 'Avg. Area Number of Bedrooms', 'Area Population', 'Price']]

#Checking for missing values and dropping them if any
print("Missing values before cleaning:")
print(housing.isnull().sum())
housing = housing.dropna()
print("Missing values after cleaning:")
print(housing.isnull().sum())

#Display summary statistics
print(housing.describe())

#Data visualization
sns.pairplot(housing)
plt.show()

#Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(housing.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

#Defining independent (X) and dependent (y) variables
x = housing[['Avg. Area Income']]
y = housing[['Area Population']]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Initializing and training the model
model = LinearRegression()
model.fit(x_train, y_train)

#Model intercept and coefficients
print("Model Intercept:", model.intercept_)
coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])
print(coeff_df)

#Making predictions
predictions = model.predict(x_test)
print("Predictions:")
print(predictions[:10])  #Display first 10 predictions

#Scatter plot: Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, edgecolor='black', alpha=0.7, color='skyblue', label='Predicted Points')

#Regression line
z = np.polyfit(y_test.values.flatten(), predictions.flatten(), 1)  #Linear fit
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='red', linewidth=2, label='Regression Line')

#Prediction line (y=x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='green', linewidth=2, label='Perfect Prediction')

#Labels and Title
plt.xlabel('Actual Area Population', fontsize=12, weight='bold')
plt.ylabel('Predicted Area Population', fontsize=12, weight='bold')
plt.title('Actual vs Predicted Area Population with Regression Line', fontsize=14, weight='bold')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()
