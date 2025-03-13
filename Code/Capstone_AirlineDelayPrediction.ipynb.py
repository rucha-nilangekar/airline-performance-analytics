# Class: ISDS 577
# CAPSTONE PROJECT

# In[10]: Reading in the data set

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/nilan/OneDrive - Cal State Fullerton/Attachments/Fall_2023/577_capstone/airline_cleaned_csv.csv')

# print(df)
column_names = df.columns.tolist()
# Print all variable names
print('All Columns Names:')
print(column_names)


# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object'])

# Print the names of categorical columns
print(' ')
print(' ')
print('Categorical Columns Names:')
print(categorical_columns.columns.tolist())


# Identify numerical columns 
numerical_columns = df.select_dtypes(include=['number'])

# Print the names of numerical columns
print(' ')
print(' ')
print('Numerical Columns Names:')
print(numerical_columns.columns.tolist())



# In[20]: Cleaning the missing values
import seaborn as sns
import matplotlib.pyplot as plt

# Create a boolean mask for missing values (True for missing, False for non-missing)
missing_mask = df.isna()  # You can also use df.isnull()

# Use seaborn's heatmap to visualize the missing value map
plt.figure(figsize=(8, 6))
sns.heatmap(missing_mask, cmap='viridis', cbar=False)
plt.title('Missing Value Map')
plt.show()


# Create a boolean mask for missing values (True for missing, False for non-missing)
missing_mask = df.isna()  # You can also use df.isnull()

# Count the number of rows with missing values
num_rows_with_missing = missing_mask.any(axis=1).sum()

print(f"Number of rows with missing values: {num_rows_with_missing}")

#remove all rows with missing values
df = df.dropna()

print(df)

#clean up the Variable Explorer
del missing_mask
del num_rows_with_missing
del column_names



# In[30]: Heat map for numerical variables


# corellation analysis for numerical variables
import seaborn as sns
import matplotlib.pyplot as plt

#numerical dataframe
num_df = df[['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'DOT_ID_Reporting_Airline', 'Flight_Number_Reporting_Airline', 'OriginAirportID', 'DestAirportID', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes', 'DepDel15', 'DepartureDelayGroups', 'TaxiOut', 'TaxiIn', 'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups', 'Cancelled', 'CRSElapsedTime', 'ActualElapsedTime', 'Distance', 'DistanceGroup']]

correlation_matrix = num_df.corr()  
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()


# Set the correlation threshold
threshold = 0.7  # You can change this threshold as needed

# Create empty lists to store variable pairs with strong correlations
strong_positive_correlations = []
strong_negative_correlations = []

# Loop through the correlation matrix and identify strong correlations
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            if correlation_matrix.iloc[i, j] > 0:
                strong_positive_correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
            else:
                strong_negative_correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# Print or display the results
print("Strong Positive Correlations:")
for variable1, variable2, correlation in strong_positive_correlations:
    print(f"{variable1} and {variable2}: {correlation:.2f}")

print("\nStrong Negative Correlations:")
for variable1, variable2, correlation in strong_negative_correlations:
    print(f"{variable1} and {variable2}: {correlation:.2f}")



# In[41]: Dropping numerical variables


# Drop a variable (column) by specifying its name
columns_to_drop = ['Quarter', 'DepDelay', 'DepDel15', 'DepartureDelayGroups','ArrDelay','ArrDel15','ArrivalDelayGroups','ActualElapsedTime','Distance','DistanceGroup']
df = df.drop(columns=columns_to_drop)

print(df)




# In[50]: Checking corellation between 2 categorical variables


# check corellation of possible multicollinearity for Categorical Variables

from scipy.stats import chi2_contingency

# Create a contingency table of the two categorical variables
contingency_table = pd.crosstab(df['IATA_CODE_Reporting_Airline'], df['Reporting_Airline'])

# Calculate chi-square statistic and p-value
chi2, p, _, _ = chi2_contingency(contingency_table)

# Print results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")



# since the p value is less than alpha (0.05) there are multicollinearity between these 2 categorical variable. 
# Therefore, I'll only keep Reporting_Airline
# Since we're analyzing the delays, cancellation will not be in the interest of this quesion so I'll also drop Cancellation Code

# Categorical Variable Column Names:
# FlightDate
# Reporting_Airline
# IATA_CODE_Reporting_Airline
# Tail_Number
# Origin
# OriginState
# Dest
# DestState
# DepTimeBlk
# ArrTimeBlk
# CancellationCode
# CarrierDelay
# WeatherDelay
# NASDelay
# SecurityDelay
# LateAircraftDelay


# In[51]: Dropping categorical variables


# Drop a variable (column) by specifying its name
columns_to_drop = ['IATA_CODE_Reporting_Airline', 'CancellationCode', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df = df.drop(columns=columns_to_drop)

print(df)



# In[60]: Dummy Variables for Categorical Variables

# Shuffling the DataFrame
shuffled_df = df.sample(frac=1)

print(shuffled_df)

# Selecting the categorical columns
categorical_columns_names = ['FlightDate', 'Reporting_Airline', 'Tail_Number', 'Origin', 'OriginState', 'Dest', 'DestState', 'DepTimeBlk', 'ArrTimeBlk']

# Using get_dummies() to convert categorical variable into dummy/indicator variables
dummies = pd.get_dummies(shuffled_df, columns=categorical_columns_names)



# In[70]: Corellation for Categorical variables
# del numerical_columns
# del categorical_columns
# del df_drop
# del dummies

# dummies = dummies.astype('float32')

# Creating a new DataFrame with n lines
dummy_subset = dummies.head(6500)


# Compute the correlation matrix
correlation_matrix = dummy_subset.corr()

# Display the correlation matrix
# print("Correlation matrix for dummy variables:")
# print(correlation_matrix)

print("Categories with correlation greater than 0.7:")
for col in correlation_matrix.columns:
    for index, value in correlation_matrix[col].items():
        if col != index and abs(value) > 0.7:
            print(f"{col} - {index}")

# In[71]: Dropping categorical variables


# Drop a variable (column) by specifying its name
columns_to_drop = ['OriginState', 'DestState', 'Tail_Number']
df = df.drop(columns=columns_to_drop)

print(df)


############################### END DATA PREP #################################








# In[00]:

########################### RESTART SHORT CUT #################################
# Because I already did all of the data preprocessing, I do not want to run all the code again. 
# I just need to remove all of the columns I don't need, create dummy variables and start running the algorithm


# In[01]: Prepare dummy


import pandas as pd
import numpy as np

# Loading data
df = pd.read_csv('D:/CSUF/2023 Fall/ISDS 577/Final Project/airline_cleaned_csv.csv')

import seaborn as sns
import matplotlib.pyplot as plt

# Cutting the data in half
df = df[df['Year'] >= 2015]

print(df)


# Drop a variable (column) by specifying its name
columns_to_drop = ['Quarter', 'DepDelay', 'DepDelayMinutes', 'DepDel15', 'DepartureDelayGroups', 'ArrDelay', 'ArrDel15', 'ArrivalDelayGroups','ActualElapsedTime','Distance','DistanceGroup', 'IATA_CODE_Reporting_Airline', 'CancellationCode', 'OriginState', 'DestState', 'Tail_Number', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df = df.drop(columns=columns_to_drop)

# Create a boolean mask for missing values (True for missing, False for non-missing)
missing_mask = df.isna()  # You can also use df.isnull()

# Use seaborn's heatmap to visualize the missing value map
plt.figure(figsize=(8, 6))
sns.heatmap(missing_mask, cmap='viridis', cbar=False)
plt.title('Missing Value Map')
plt.show()


# Create a boolean mask for missing values (True for missing, False for non-missing)
missing_mask = df.isna()  # You can also use df.isnull()

# Count the number of rows with missing values
num_rows_with_missing = missing_mask.any(axis=1).sum()

print(f"Number of rows with missing values: {num_rows_with_missing}")

#remove all rows with missing values
df = df.dropna()

print(df)

#clean up the Variable Explorer
del missing_mask
del num_rows_with_missing



print(df)


# Selecting the categorical columns
categorical_columns_names = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 'Reporting_Airline', 'Origin', 'Dest', 'DepTimeBlk', 'ArrTimeBlk']

# Using get_dummies() to convert categorical variable into dummy/indicator variables
dummies = pd.get_dummies(df, columns=categorical_columns_names)

# print(df)
column_names = dummies.columns.tolist()
# Print all variable names
print('All Columns Names:')
print(column_names)



del categorical_columns_names
del columns_to_drop


print(dummies)

del df


# In[00]:
############################ DATA ANALYSIS BEGIN ##############################



# In[10]: Regression Analysis
# del numerical_columns
# del categorical_columns
# del categorical_columns_names
# del df_drop
# del only_dummy_cat
# del shuffled_df
# del columns_to_drop


from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Dataset
X = dummies.drop(columns=['ArrDelayMinutes'], axis=1)
y = dummies['ArrDelayMinutes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the SGDRegressor model
model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)


# Extract and display the significant variables
coefficients = pd.Series(model.coef_, index=X.columns)
significant_variables = coefficients[coefficients.abs() > 0.01]  # Alpha = 1%
significant_variables_sorted = significant_variables.abs().sort_values(ascending=False)
print("Most significant variables:")
print(significant_variables_sorted)


# Evaluating the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")






# In[20]: Decision Tree 

# Importing required libraries

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error


# Dataset
X = dummies.drop(columns=['ArrDelayMinutes'])
y = dummies['ArrDelayMinutes']


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a decision tree regressor
regressor = DecisionTreeRegressor()

# Training the decision tree regressor
regressor.fit(X_train, y_train)

# Making predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluating the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")





# In[30]: Random forest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # For regression tasks
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Dataset
X = dummies.drop(columns=['ArrDelayMinutes'])
y = dummies['ArrDelayMinutes']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create a Random Forest Regressor (or Classifier) instance
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)  

# Fit the model on the training data
model.fit(X_train, y_train)


# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Evaluating the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



################CHOOSING THE n-estimator ######################################


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # Dataset
# X = dummies.drop(columns=['ArrDelayMinutes'])
# y = dummies['ArrDelayMinutes']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Create a Random Forest Regressor instance
# model = RandomForestRegressor(oob_score=True, n_jobs=-1, random_state=42)

# # Fit the model on the training data
# model.fit(X_train, y_train)

# # Obtain the OOB (Out-of-Bag) error for different number of estimators.
# oob_errors = []
# for n_trees in range(1, 5):
#     model.set_params(n_estimators=n_trees)
#     model.fit(X_train, y_train)
#     oob_errors.append(1 - model.oob_score_)

# # Find the optimal number of estimators with the lowest OOB error
# optimal_n_estimators = np.argmin(oob_errors) + 1
# print(f"Optimal number of estimators: {optimal_n_estimators}")

# # Retrain the model with the optimal number of estimators
# model.set_params(n_estimators=optimal_n_estimators)
# model.fit(X_train, y_train)

# # Predict the target variable for the test set
# y_pred = model.predict(X_test)

# # Evaluating the model
# print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# # Calculate the mean squared error
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")






# In[00]:

################################ DEPARTURE ####################################



# In[01]: Prepare dummy
############################# FOR DEPARTURE DELAY #############################

import pandas as pd
import numpy as np

# Loading data
df = pd.read_csv('D:/CSUF/2023 Fall/ISDS 577/Final Project/airline_cleaned_csv.csv')

import seaborn as sns
import matplotlib.pyplot as plt

# Cutting the data in half
df = df[df['Year'] >= 2015]

print(df)


# Drop a variable (column) by specifying its name
columns_to_drop = ['Quarter', 'DepDelay', 'DepDel15', 'DepartureDelayGroups', 'ArrDelay', 'ArrDel15', 'ArrDelayMinutes', 'ArrivalDelayGroups','ActualElapsedTime','Distance','DistanceGroup', 'IATA_CODE_Reporting_Airline', 'CancellationCode', 'OriginState', 'DestState', 'Tail_Number', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df = df.drop(columns=columns_to_drop)

# Create a boolean mask for missing values (True for missing, False for non-missing)
missing_mask = df.isna()  # You can also use df.isnull()

# Use seaborn's heatmap to visualize the missing value map
plt.figure(figsize=(8, 6))
sns.heatmap(missing_mask, cmap='viridis', cbar=False)
plt.title('Missing Value Map')
plt.show()


# Create a boolean mask for missing values (True for missing, False for non-missing)
missing_mask = df.isna()  # You can also use df.isnull()

# Count the number of rows with missing values
num_rows_with_missing = missing_mask.any(axis=1).sum()

print(f"Number of rows with missing values: {num_rows_with_missing}")

#remove all rows with missing values
df = df.dropna()

print(df)

#clean up the Variable Explorer
del missing_mask
del num_rows_with_missing



print(df)


# Selecting the categorical columns
categorical_columns_names = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 'Reporting_Airline', 'Origin', 'Dest', 'DepTimeBlk', 'ArrTimeBlk']

# Using get_dummies() to convert categorical variable into dummy/indicator variables
dummies = pd.get_dummies(df, columns=categorical_columns_names)

# print(df)
column_names = dummies.columns.tolist()
# Print all variable names
print('All Columns Names:')
print(column_names)



del categorical_columns_names
del columns_to_drop


print(dummies)

del df


# In[00]:
############################ DATA ANALYSIS BEGIN ##############################



# In[10]: Regression Analysis
# del numerical_columns
# del categorical_columns
# del categorical_columns_names
# del df_drop
# del only_dummy_cat
# del shuffled_df
# del columns_to_drop



from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Dataset
X = dummies.drop(columns=['DepDelayMinutes'], axis=1)
y = dummies['DepDelayMinutes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the SGDRegressor model
model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)


# Extract and display the significant variables
coefficients = pd.Series(model.coef_, index=X.columns)
significant_variables = coefficients[coefficients.abs() > 0.01]  # Alpha = 1%
significant_variables_sorted = significant_variables.abs().sort_values(ascending=False)
print("Most significant variables:")
print(significant_variables_sorted)


# Evaluating the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")






# In[10]: Decision Tree 

# Importing required libraries

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt #visualizing the tree


# Dataset
X = dummies.drop(columns=['DepDelayMinutes'])
y = dummies['DepDelayMinutes']


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a decision tree regressor
regressor = DecisionTreeRegressor()

# Training the decision tree regressor
regressor.fit(X_train, y_train)

# Making predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluating the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")




# In[10]: Random forest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # For regression tasks
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Dataset
X = dummies.drop(columns=['DepDelayMinutes'])
y = dummies['DepDelayMinutes']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create a Random Forest Regressor (or Classifier) instance
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)  

# Fit the model on the training data
model.fit(X_train, y_train)


# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Evaluating the model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

