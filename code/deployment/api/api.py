from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model
model = joblib.load('../../models/model.pkl')

# Define the input schema
class PassengerData(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int

@app.post('/predict')
def predict_survival(data: PassengerData):
    # Convert input data into numpy array
    features = np.array([[data.Pclass, data.Sex, data.Age, data.SibSp, data.Parch, data.Fare, data.Embarked]])
    # Get the model prediction
    prediction = model.predict(features)
    # Return the prediction as a response
    return {"Survived": int(prediction[0])}
