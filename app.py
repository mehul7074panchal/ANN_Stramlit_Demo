import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


model = load_model('model.h5')

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


    st.title("Customer Churn Prediction")


    geography = st.selectbox("Geography", label_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    input_data = {
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
      
    }


geo_encode = label_encoder_geo.transform([[geography]])
geo_encode_dense = geo_encode.toarray()  # Convert sparse matrix to dense array
geo_encode_df = pd.DataFrame(geo_encode_dense, columns=label_encoder_geo.get_feature_names_out(['Geography']))


input_df = pd.DataFrame(input_data)  # Convert input_data to a DataFrame
input_df = pd.concat([input_df.reset_index(drop=True), geo_encode_df], axis=1)

# input_df = pd.concat([input_df.drop(['Geography'], axis=1), geo_encode_df], axis=1)
# input_df = pd.concat([input_data.reset_index(drop = True).drop(['Geography']), geo_encode_df], axis=1)

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_prod = prediction[0][0]
if prediction_prod > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
    # st.write("Input Data:")
    # st.write(input_df)
    # st.write("Scaled Input Data:")
    # st.write(input_scaled)
    st.write("Prediction Result:")
    st.write(f'{prediction_prod:.2f}')
    # st.write("Model Summary:")
    # st.write(model.summary())
    # st.write("Label Encoders:")
    # st.write("Geography Label Encoder:")
    # st.write(label_encoder_geo)
