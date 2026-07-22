
# House Price Prediction 🏠

This is a Machine Learning project that predicts house prices based on different features of a house and its location.

I built this project using **Python, Scikit-learn and FastAPI**. After training the model, I saved the trained model and scaler using Pickle and integrated them into a FastAPI application.

The project also has a simple interactive frontend where users can enter the house details and get the predicted price.

## Features

* Predicts house prices using Machine Learning
* Uses Linear Regression model
* Uses StandardScaler for preprocessing
* Trained model saved as `model.pkl`
* Scaler saved as `scaler.pkl`
* FastAPI used for backend
* Simple HTML and CSS based frontend
* Prediction result is shown on the same page

## Dataset

I used the **California Housing Dataset** available in Scikit-learn.

The model uses these 8 features:

* MedInc
* HouseAge
* AveRooms
* AveBedrms
* Population
* AveOccup
* Latitude
* Longitude

The target variable is the median house value.

## Machine Learning

For this project, I used **Linear Regression** to predict the house price.

Before giving the data to the model, I used `StandardScaler` to scale the input features.

The model and scaler are saved using Pickle:

```text
model.pkl
scaler.pkl
```

During prediction, the input data is first scaled using `scaler.pkl` and then passed to the trained model.

## Project Structure

```text
House_price_prediction/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── README.md
│
└── templates/
    └── index.html
```

## How to Run

First, install the required libraries:

```bash
pip install fastapi uvicorn jinja2 python-multipart scikit-learn numpy
```

Then start the FastAPI application:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

After running the server, open the application in your browser.

```text
http://localhost:8001
```

If you are using GitHub Codespaces, you can open the application from the **PORTS** section by opening port `8001`.

## How It Works

The user enters the values of the 8 features in the frontend.

Then:

```text
User Input
    ↓
FastAPI
    ↓
StandardScaler
    ↓
Trained Model
    ↓
Predicted House Price
    ↓
Result on Web Page
```

The prediction is calculated by the trained Linear Regression model and displayed on the frontend.

## Model Evaluation

I checked the performance of the model using regression metrics such as:

* MAE
* MSE
* RMSE
* R² Score

Since this is a regression problem, I used these metrics instead of classification accuracy.

## Technologies Used

* Python
* NumPy
* Scikit-learn
* Pandas
* Linear Regression
* StandardScaler
* FastAPI
* Uvicorn
* HTML
* CSS
* Pickle

## Future Improvements

In the future, I would like to improve this project by:

* Trying other ML models like Random Forest and XGBoost
* Improving the prediction accuracy
* Making the frontend more interactive
* Deploying the project online
* Adding graphs and data visualizations

## Author

**Adarsh**

This project was created as a Machine Learning project to understand how a trained ML model can be integrated with a FastAPI application and used through a web interface.
