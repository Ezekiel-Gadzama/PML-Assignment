import streamlit as st
import requests

# Define the Streamlit web app interface
st.title("Titanic Survival Prediction")

Pclass = st.selectbox('Ticket Class (Pclass)', [1, 2, 3])
Sex = st.selectbox('Sex', ['Male', 'Female'])
Age = st.number_input('Age', min_value=0.0, max_value=100.0, value=30.0)
SibSp = st.number_input('Number of Siblings/Spouses Aboard (SibSp)', min_value=0, max_value=10, value=0)
Parch = st.number_input('Number of Parents/Children Aboard (Parch)', min_value=0, max_value=10, value=0)
Fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=32.0)
Embarked = st.selectbox('Port of Embarkation (Embarked)', ['S', 'C', 'Q'])

if st.button('Predict'):
    # Prepare the input data
    input_data = {
        "Pclass": Pclass,
        "Sex": 0 if Sex == 'Male' else 1,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": {'S': 0, 'C': 1, 'Q': 2}[Embarked]
    }
    
    # Send the input data to the FastAPI endpoint
    response = requests.post('http://fastapi:80/predict', json=input_data)
    # Display the prediction result
    prediction = response.json()
    st.write(f"Survived: {'Yes' if prediction['Survived'] == 1 else 'No'}")
