# -*- coding: utf-8 -*-
"""PA_Final_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Z4ZMNa6S94-6pNit1cUkCSsBQLJFrIqz
"""

import pandas as pd

from google.colab import files
uploaded = files.upload()

df=pd.read_parquet('/content/green_tripdata_2023-12.parquet')

df.info()

# Drop the 'ehail_fee' column from the DataFrame
df.drop(columns=['ehail_fee'], inplace=True)

# Convert pickup and dropff columns to datetime if not already
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

# Calculate trip duration in minutes
df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60

# # Extract weekday name from dropoff datetime
df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()

#Display count of trips for each weekday
df['weekday'].value_counts()

#Extract hour of the day from dropoff datetime
df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour

# Display count of trips for each hour of the day
df['hourofday'].value_counts().sort_index()

# Check how many missing values are in each column
print("Missing values before imputations:")
print(df.isnull().sum())

# Impute missing values
# For numeric colums: fill with median
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# For categorical/ object columsn: fill with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
  df[col] = df[col].fillna(df[col].mode()[0])

# Confirm missing values have been handled
print("\nMissing values after imputations:")
print(df.isnull().sum())

import matplotlib.pyplot as plt
# Pie chart for payment_type
plt.figure(figsize=(6, 6))
df['payment_type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Payment Types')
plt.ylabel('')
plt.axis('equal')  # <-- This line ensures the pie is a circle
plt.show()

# Pie chart for trip_type
plt.figure(figsize=(6, 6))
df['trip_type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Trip Types')
plt.ylabel('')
plt.axis('equal')
plt.show()

# Group by weekday and calculate average total_amount
avg_amount_by_weekday = df.groupby('weekday')['total_amount'].mean()

# Display the result
print(avg_amount_by_weekday)

# Group by payment_type and calculate average total_amount
avg_amount_by_payment = df.groupby('payment_type')['total_amount'].mean()


#Display the result
print(avg_amount_by_payment)

#j)	Groupby() of average tip_amount & weekday
avg_tip_by_weekday =df.groupby('weekday')['tip_amount'].mean()

#Display the result
print(avg_tip_by_weekday)

#k)	Groupby() of average tip_amount & payment_type
avg_tip_by_payment = df.groupby('payment_type')['tip_amount'].mean()

#Display the result
print(avg_tip_by_payment)

#l)	Test null average total_amount of different trip_type is identical
from scipy.stats import f_oneway

#Drop rows where total_amount or trip_type is null
filtered_df = df[['trip_type', 'total_amount']].dropna()

#Separate total_amount values based on trip_type
trip_type_groups = [group['total_amount'].values for name, group in filtered_df.groupby('trip_type')]

# Perform one-way ANOVA test
f_stat, p_value = f_oneway(*trip_type_groups)

#Display result
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

#Interpretation
if p_value < 0.05:
   print("Reject the null hypothesis: Average total_amount differs by trip_type.")
else:
   print("Fail to reject the null hypothesis: Average total_amount is similar across trip_type.")

#m)	Test null average total_amount of different weekday is identical

from scipy.stats import f_oneway

# Drop rows with missing values in weekday or total_amount
filtered_df = df[['weekday', 'total_amount']].dropna()

# Group total_amount by weekday
weekday_groups = [group['total_amount'].values for name, group in filtered_df.groupby('weekday')]

# Perform one-way ANOVA
f_stat, p_value = f_oneway(*weekday_groups)

# Display results
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: Average total_amount differs by weekday.")
else:
    print("Fail to reject the null hypothesis: Average total_amount is similar across weekdays.")

from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table = pd.crosstab(df['trip_type'], df['payment_type'])

# Perform Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Display results
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
print("\nExpected Frequencies Table:")
print(expected)

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: There is an association between trip_type and payment_type.")
else:
    print("Fail to reject the null hypothesis: No association between trip_type and payment_type.")

#o)	Numeric variables are trip_distance, fare_amount, extra, mta_tax, tip_amount, tolls_amount, improvemnet_surcharge, congestion_surcharge, trip_duration, passenger_count
# Define list of numeric variables
numeric_vars = [
    'trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
    'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
    'trip_duration', 'passenger_count'
]

# Display summary statistics
df[numeric_vars].describe()

#p)	Object variables are store and fwd flag, RatecodeID, payment_type, trip_type, weekday and hourof day
# Define list of object (categorical) variables
categorical_vars = [
    'store_and_fwd_flag', 'RatecodeID', 'payment_type',
    'trip_type', 'weekday', 'hourofday'
]

# Display the unique values in each categorical variable
for col in categorical_vars:
    print(f"\n{col} - Unique Values:")
    print(df[col].unique())

#q)	Correlation analysis of numeric cols
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix for numeric columns
corr_matrix = df[numeric_vars].corr()

# Display correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numeric Variables")
plt.show()

#r)	Dummy encode object cols
# Dummy encode the categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Display the new shape of the encoded dataframe
print("Shape after encoding:", df_encoded.shape)
df_encoded.head()

#s)	Dependent Variable is total_amount.  Histogram, Boxplot and Density Curve of this variable
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Create a figure with subplots
plt.figure(figsize=(18, 5))

# Histogram
plt.subplot(1, 3, 1)
sns.histplot(df['total_amount'], kde=False, bins=40, color='skyblue')
plt.title("Histogram of Total Amount")
plt.xlabel("Total Amount")
plt.ylabel("Frequency")

# Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(x=df['total_amount'], color='lightgreen')
plt.title("Boxplot of Total Amount")
plt.xlabel("Total Amount")

# Density Curve (KDE)
plt.subplot(1, 3, 3)
sns.kdeplot(df['total_amount'], fill=True, color='salmon')
plt.title("Density Curve of Total Amount")
plt.xlabel("Total Amount")

plt.tight_layout()
plt.show()

#t)	Build the following Regression Models:
#Multiple Linear Regression
#Decision Tree
#Random Forest with 100 trees
#Gradient Boosting with 100 trees

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Drop datetime columns from df_encoded
datetime_cols = df_encoded.select_dtypes(include='datetime64').columns
df_model = df_encoded.drop(columns=datetime_cols)

# Step 2: Define features (X) and target (y)
X = df_model.drop('total_amount', axis=1)
y = df_model['total_amount']

# Step 3: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest (100 trees)': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting (100 trees)': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Step 5: Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

