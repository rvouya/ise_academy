import streamlit as st
import pandas as pd
import pickle

from huggingface_hub import login

# Use your Hugging Face token
login(token="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")


model = pickle.load(open('C:\\dasprog well\\fp_ise\\model.pkl', 'rb'))

from huggingface_hub import create_repo

# Replace 'your_model_name' with the name you want for your model
repo_url = create_repo(name='Almond Classification', private=False)

st.title('Almond Classification')
st.write('This web app classifies almonds based on your input features.')


# Input untuk setiap fitur
length_major_axis = st.number_input('Length (major axis)', min_value=0.0)
width_minor_axis = st.number_input('Width (minor axis)', min_value=0.0)
thickness_depth = st.number_input('Thickness (depth)', min_value=0.0)
area = st.number_input('Area', min_value=0.0)
perimeter = st.number_input('Perimeter', min_value=0.0)
roundness = st.slider('Roundness', min_value=0.0, max_value=1.0, step=0.01)
solidity = st.slider('Solidity', min_value=0.0, max_value=1.0, step=0.01)
compactness =  st.slider('Compactness', min_value=0.0, max_value=1.0, step=0.01)
aspect_ratio =  st.slider('Aspect Ratio', min_value=0.0, max_value=5.0, step=0.01)
eccentricity = st.slider('Eccentricity', min_value=0.0, max_value=1.0, step=0.01)
extent = st.slider('Extent', min_value=0.0, max_value=1.0, step=0.01)
convex_area = st.number_input('Convex hull (convex area)', min_value=0.0, step=0.01)


# Tombol untuk memprediksi
if st.button('Predict'):
    input_features = [[length_major_axis, width_minor_axis, thickness_depth, area,
                       perimeter, roundness, solidity, compactness, aspect_ratio,
                       eccentricity, extent, convex_area]]
    prediction = model.predict(input_features)
    st.write(f'The predicted class is: {prediction[0]}')
