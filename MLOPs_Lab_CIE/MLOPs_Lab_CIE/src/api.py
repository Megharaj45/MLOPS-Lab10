from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# Load model
model = joblib.load("models/best_model.pkl")

app = FastAPI()

# Input schema with validation
class InputData(BaseModel):
    patients_ahead: int = Field(..., ge=1, le=60)
    staff_count: int = Field(..., ge=1, le=20)
    is_emergency: int = Field(..., ge=0, le=1)
    dept_load: int = Field(..., ge=1, le=10)

# Health endpoint
@app.get("/heartbeat")
def heartbeat():
    return {"alive": True, "service": "MediQueue wait_time_min API"}

# Prediction endpoint
@app.post("/estimate")
def estimate(data: InputData):
    input_data = [[
        data.patients_ahead,
        data.staff_count,
        data.is_emergency,
        data.dept_load
    ]]
    prediction = model.predict(input_data)[0]
    return {"prediction": float(prediction)}