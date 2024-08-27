import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_set = 'test_data.xlsx'

#load dataset

data_df = pd.read_excel(data_set, sheet_name = 'data')
labels_df = pd.read_excel(data_set, sheet_name = 'labels')


#preprocess data

#temporarily remove sample column
temp_drop_column = data_df['Sample']
data_df = data_df.drop(columns=['Sample']).apply(pd.to_numeric, errors = 'coerce')

#Define the condition for anomalies
condition = (data_df > 99) | (data_df.isna())

#Mask the bad values and calculate the mean of the good values
good_values_df = data_df.where(~condition)

#Calculate the mean of all remaining good values
mean_value = good_values_df.mean().mean()

#Round the calculated mean. replace the bad values with the rounded mean

rounded_mean_value = round(mean_value)
data_df = data_df.where(~condition, rounded_mean_value)

# Add the Sample column back
data_df['Sample'] = temp_drop_column

#Split the data 
x = data_df.drop(columns=['Sample'])
y = labels_df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 50)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

y_prediction = model.predict(x_test_scaled)

print(classification_report(y_test, y_prediction))
