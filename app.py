from pathlib import Path
import pickle

import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = FastAPI()
env = Environment(loader=FileSystemLoader("templates"))

MODEL_PATH = Path("model.pkl")
SCALER_PATH = Path("scaler.pkl")


def train_and_save_model():
    X, y = make_regression(
        n_samples=2000,
        n_features=8,
        noise=25,
        random_state=42,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    with SCALER_PATH.open("wb") as f:
        pickle.dump(scaler, f)

    return model, scaler


def load_model_and_scaler():
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
        with SCALER_PATH.open("rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    return train_and_save_model()


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