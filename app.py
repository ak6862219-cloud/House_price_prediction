from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
import pickle
import numpy as np

app = FastAPI()

env = Environment(loader=FileSystemLoader("templates"))

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    template = env.get_template("index.html")
    return HTMLResponse(content=template.render())


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    AveOccup: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...)
):

    input_data = np.array([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude,
        0.0  # placeholder for 'PRICE' column present in scaler/model training
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Dataset target is in $100,000 units
    price = prediction * 100000

    template = env.get_template("index.html")
    return HTMLResponse(content=template.render(
        prediction=round(float(prediction), 4),
        price=f"${price:,.2f}"
    ))