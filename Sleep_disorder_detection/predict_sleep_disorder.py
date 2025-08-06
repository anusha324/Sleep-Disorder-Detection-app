import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and encoders
model = joblib.load('xgb_sleep_model.pkl')
scaler = joblib.load('scaler.pkl')
bmi_encoder = joblib.load('BMI Category_encoder.pkl')
gender_encoder = joblib.load('Gender_encoder.pkl')
disorder_mapping = joblib.load('sleep_disorder_mapping_encoder.pkl')
reverse_disorder_mapping = {v: k for k, v in disorder_mapping.items()}

# List of features in order (including Sleep Efficiency)
feature_names = [
    'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Sleep_Efficiency'
]

# Get user input
print('Enter the following details:')
user_input = {}
user_input['Gender'] = input('Gender (Male/Female): ')
user_input['Age'] = int(input('Age: '))
user_input['Sleep Duration'] = int(input('Sleep Duration (hours): '))
user_input['Quality of Sleep'] = int(input('Quality of Sleep (1-10): '))
user_input['Physical Activity Level'] = int(input('Physical Activity Level (minutes): '))
user_input['Stress Level'] = int(input('Stress Level (1-10): '))
user_input['BMI Category'] = input('BMI Category (Normal/Overweight): ')
user_input['Heart Rate'] = int(input('Heart Rate (bpm): '))
user_input['Daily Steps'] = int(input('Daily Steps: '))

# Calculate Sleep Efficiency automatically (Quality of Sleep / Sleep Duration)
user_input['Sleep_Efficiency'] = user_input['Quality of Sleep'] / user_input['Sleep Duration']
user_input['Sleep_Efficiency'] = np.clip(user_input['Sleep_Efficiency'], 0.1, 2.0)  # Clip outliers

print(f'\nCalculated Sleep Efficiency: {user_input["Sleep_Efficiency"]:.3f}')

# Encode categorical features
user_input['BMI Category'] = bmi_encoder.transform([user_input['BMI Category']])[0]
user_input['Gender'] = gender_encoder.transform([user_input['Gender']])[0]

# Arrange features in correct order for model (10 features including Sleep Efficiency)
X_input = np.array([
    user_input['Gender'],
    user_input['Age'],
    user_input['Sleep Duration'],
    user_input['Quality of Sleep'],
    user_input['Physical Activity Level'],
    user_input['Stress Level'],
    user_input['BMI Category'],
    user_input['Heart Rate'],
    user_input['Daily Steps'],
    user_input['Sleep_Efficiency']
]).reshape(1, -1)

# Scale features
X_input_scaled = scaler.transform(X_input)

# Predict
pred = model.predict(X_input_scaled)[0]

# Output result
print(f'\nPredicted Sleep Disorder: {reverse_disorder_mapping[pred]}') 