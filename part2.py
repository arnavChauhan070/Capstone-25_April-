from fastapi import FastAPI
from pydantic import BaseModel
import requests
import logging


ML_ENDPOINT = "ENDPOINT"
ML_API_KEY = "API_KEY"

REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ML_API_KEY}"
}


logging.basicConfig(
    filename="app_monitor.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


app = FastAPI(title="Machine Health Monitoring API")


class MachineInput(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    humidity: float = 50.0
    timestamp: str = "2024-05-01"
    machine_id: str = "M001"


def check_anomaly(sensor: MachineInput):
    issues = []

    if sensor.temperature > 100:
        issues.append("Temperature too high")
    if sensor.vibration > 1.5:
        issues.append("Excessive vibration")
    if sensor.pressure > 80:
        issues.append("Pressure overload")

    return ", ".join(issues) if issues else None


@app.get("/")
def root():
    return {"message": "Machine Monitoring Service is Active"}


@app.post("/predict")
def predict(sensor: MachineInput):
    try:

        request_body = {
            "Inputs": {
                "input1": [{
                    "Temperature": sensor.temperature,
                    "Vibration": sensor.vibration,
                    "Pressure": sensor.pressure,
                    "Humidity": sensor.humidity,
                    "Timestamp": sensor.timestamp,
                    "MachineID": sensor.machine_id
                }]
            },
            "GlobalParameters": {}
        }

       
        api_response = requests.post(
            ML_ENDPOINT,
            json=request_body,
            headers=REQUEST_HEADERS
        )

        response_json = api_response.json()

      
        if isinstance(response_json, list):
            prob = response_json[0].get("Scored Probabilities", 0)

        elif "Results" in response_json:
            prob = response_json["Results"]["WebServiceOutput0"][0]["Scored Labels"]

        else:
            prob = response_json.get("prediction", 0)

        prob = float(prob)

        
        anomaly_info = check_anomaly(sensor)


        logging.info(f"Request: {request_body} | Prediction: {prob}")

     
        if prob > 0.8 or anomaly_info:
            alert = f"ALERT -> Probability: {prob}, Issue: {anomaly_info}"
            print(alert)
            logging.warning(alert)

        return {
            "status": "ok",
            "failure_probability": round(prob, 2),
            "anomaly": anomaly_info
        }

    except Exception as err:
        logging.error(f"Error: {str(err)}")
        return {
            "status": "failed",
            "error": str(err)
        }