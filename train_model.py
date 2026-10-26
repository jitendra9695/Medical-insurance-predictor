import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
  os.makedirs('models')

# Loading data
df = pd.read_csv('data/insurance.csv')

# encoding the categorical data
encoding_maps = {
  'sex' : {'male' : 0, 'female' : 1},
  'smoker' : {'yes':0, 'no':1},
  'region' : {'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}
}

df.replace(encoding_maps, inplace=True)

# Split Features and Target
X = df.drop(columns='charges', axis=1)
Y = df['charges']

# Train Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
print("Training model....")
model = LinearRegression()
model.fit(X_train, Y_train)

# Evaluate 
r2 = model.score(X_test, Y_test)
print(f"Model R2 Score: {r2:4f}")

# Saving the model
model_data = {
  "model": model,
  "encoders": encoding_maps,
  "features_name": X.columns.tolist()
}

joblib.dump(model_data,'models/insurance_model.pkl')
print("Model saved to models/insurance_model.pkl")