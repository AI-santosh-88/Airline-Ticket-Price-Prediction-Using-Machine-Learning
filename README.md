# Title: Airline Ticket Price Prediction Using Machine Learning

## Description:
* This project focuses on developing a predictive model to estimate airline ticket prices based on various factors. By analyzing a dataset of flight details, including airline, source and destination cities, departure and arrival times, number of stops, class, duration, and days left before departure, the model aims to provide accurate price predictions. This information can be valuable for both travelers seeking to find the best deals and airlines looking to optimize pricing strategies.

## Responsibilities:
#### 1.Data Acquisition and Cleaning:
* Importing and loading the airline ticket price dataset.
* Identifying and handling missing values and outliers.
* Removing irrelevant columns and ensuring data consistency.
  
#### 2.Exploratory Data Analysis (EDA):
* Performing statistical analysis to understand data distribution and relationships.
* Visualizing data using matplotlib and seaborn to identify trends and patterns.
* Analyzing the impact of various features (airline, class, stops, time, city, duration, days left) on ticket prices.

#### 3.Feature Engineering:
* Converting categorical variables into numerical representations using Label Encoding.
* Scaling numerical features using MinMaxScaler to improve model performance.
  
#### 4.Model Building and Evaluation:
* Splitting the dataset into training and testing sets.
* Training and evaluating various regression models, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, KNeighbors Regressor, Extra Trees Regressor, Gradient 
  Boosting Regressor, XGBRegressor, Bagging Regressor, Ridge Regression, and Lasso Regression.
* Calculating and comparing evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R2 score, Adjusted R2 score, and Mean Absolute 
  Percentage Error (MAPE).
* Selecting the best model based on the evaluation metrics.
* Visualizing the actual vs predicted prices.

#### 5.Model Deployment (Potential Future Step):
* Saving the trained model for future use.
* Developing a user interface or API for price prediction.

## Libraries Used:
#### 1.pandas:
* For data manipulation and analysis.
#### 2.numpy: 
* For numerical computations.
#### 3.matplotlib: 
* For creating static, interactive, and animated visualizations.1   
#### 4.seaborn: 
* For enhanced data visualization.
#### 5.scikit-learn (sklearn): 
* For machine learning algorithms, preprocessing, and model evaluation.
#### 6.xgboost: 
* For gradient boosting framework.
#### 7.warnings: 
* to ignore warnings.




