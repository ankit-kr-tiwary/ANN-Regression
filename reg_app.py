import streamlit as st 
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
import pandas as pd 
import pickle

model = tf.keras.models.load_model('regression_model.h5')

## Loa d the encoder and scaler 

#rb = read byte mode 
with open ('label_encode_gender.pkl','rb') as file:
    label_coder_gender = pickle.load(file)

with open ('onehot_geo.pkl','rb') as file:
    onehot_geo =pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)



### Streamlit app

st.title('Estimated Salary  Predictions')

geography= st.selectbox ('Geography', onehot_geo.categories_[0]) 
gender = st.selectbox('Gender',label_coder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited =st.selectbox('Exited',[0, 1])
tenure = st.slider('Tenure',0, 10)
num_of_products = st.slider ('Number of products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_data = pd.DataFrame ({
   'CreditScore' : [credit_score],
   'Geography' : [geography],
   'Gender' : [label_coder_gender.transform([gender])[0]],
   'Age':[age],
   'Tenure':[tenure],
   'Balance' : [balance],
   'NumOfProducts' : [num_of_products],
   'HasCrCard' : [has_cr_card],
   'IsActiveMember': [is_active_member],
   'Exited':[exited]

}
)

# One-hot encode 'Geography'
geo_encoded = onehot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded column 
input_df = pd.concat([input_data.drop("Geography",axis=1),geo_encoded_df],axis=1)

## Combine one-hot encoded columns with input data 
input_scaled = scaler.transform(input_df)

# Predicted salary 
 
Prediction = model.predict(input_scaled)
Predicted_salary = Prediction[0][0]

st.write(f'Predicted Estimated Salary : {Predicted_salary:.2f}')


