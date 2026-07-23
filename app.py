from pathlib import Path
import pickle

import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

app = FastAPI()
env = Environment(loader=FileSystemLoader("templates"))

MODEL_PATH = Path("model.pkl")
SCALER_PATH = Path("scaler.pkl")


def load_model_and_scaler():
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_model_and_scaler()


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
    Longitude: float = Form(...),
):
    input_data = np.array(
        [[
            MedInc,
            HouseAge,
            AveRooms,
            AveBedrms,
            Population,
            AveOccup,
            Latitude,
            Longitude,
        ]],
        dtype=float,
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    price = float(prediction) * 100000

    template = env.get_template("index.html")
    return HTMLResponse(
        content=template.render(
            prediction=round(float(prediction), 4),
            price=f"${price:,.2f}",
        )
    )