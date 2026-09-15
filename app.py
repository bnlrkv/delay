import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the features below to predict delivery delay.')

# Input fields for features (based on x_train/x_test columns)
# You might need to adjust these based on the actual feature ranges and types
delivery_distance = st.slider('Delivery Distance', min_value=0.0, max_value=100.0, value=50.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1-5)', min_value=1, max_value=5, value=3)
delivery_slot = st.slider('Delivery Slot (1-3)', min_value=1, max_value=3, value=2)
preparation_time = st.slider('Preparation Time (minutes)', min_value=0, max_value=120, value=60)
vehicle_condition = st.slider('Vehicle Condition (1-5)', min_value=1, max_value=5, value=3)
customer_satisfaction = st.slider('Customer Satisfaction (1-5)', min_value=1, max_value=5, value=3)
rush_hour = st.selectbox('Rush Hour', [0, 1]) # Assuming 0 for no rush hour, 1 for rush hour
feature_1 = st.slider('Feature 1', min_value=0.0, max_value=200.0, value=100.0)
feature_2 = st.slider('Feature 2', min_value=0.0, max_value=50.0, value=25.0)
feature_3 = st.slider('Feature 3', min_value=0.0, max_value=200.0, value=100.0)

# Create a DataFrame for the input features
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Preparation_Time': preparation_time,
    'Vehicle_Condition': vehicle_condition,
    'Customer_Satisfaction': customer_satisfaction,
    'Rush_Hour': rush_hour,
    'Feature_1': feature_1,
    'Feature_2': feature_2,
    'Feature_3': feature_3
}])

if st.button('Predict Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]
    
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f"The model predicts a **DELAY** in delivery (Probability: {prediction_proba[1]:.2f}).")
    else:
        st.success(f"The model predicts **NO DELAY** in delivery (Probability: {prediction_proba[0]:.2f}).")
    
   


