import codecs
import csv
import logging
import pickle
import re
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def load_model_data():
    with open("model_data.pickle", "rb") as f:
        model_data = pickle.load(f)

    return model_data


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """."""
    def get_value(x: str):
        """."""
        if not pd.notnull(x):
            return None

        value = re.search(r'\d+.\d+|\d+', x)
        if not value:
            return None

        return float(value.group())

    logging.info("Preprocessing DataFrame.")

    for col in ["mileage", "engine", "max_power"]:
        df[col] = df[col].apply(get_value).astype(float)

    df["name"] = df["name"].apply(lambda x: x.split(' ')[0])

    categorical = df[["name", "fuel", "seller_type", "transmission", "owner"]]
    numerical = df.drop(columns=categorical.columns)

    numerical["power_per_volume"] =  numerical["max_power"] /   numerical["engine"]
    numerical["year_square"] = numerical["year"]**2
    numerical["engine_log"] = np.log(numerical["engine"])
    numerical["km_driven_log"] = np.log(numerical["km_driven"])

    numerical = pd.DataFrame(MODEL_DATA["StSc"].transform(numerical), columns=numerical.columns)

    categorical = pd.DataFrame(MODEL_DATA["OHE"].transform(categorical).toarray(),
                               columns=MODEL_DATA["OHE"].get_feature_names_out())
    
    df = pd.concat([numerical, categorical], axis=1)

    assert (df.columns.values == np.array(
        ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
       'power_per_volume', 'year_square', 'engine_log', 'km_driven_log',
       'name_Ashok', 'name_Audi', 'name_BMW', 'name_Chevrolet',
       'name_Daewoo', 'name_Datsun', 'name_Fiat', 'name_Force',
       'name_Ford', 'name_Honda', 'name_Hyundai', 'name_Isuzu',
       'name_Jaguar', 'name_Jeep', 'name_Kia', 'name_Land', 'name_Lexus',
       'name_MG', 'name_Mahindra', 'name_Maruti', 'name_Mercedes-Benz',
       'name_Mitsubishi', 'name_Nissan', 'name_Opel', 'name_Peugeot',
       'name_Renault', 'name_Skoda', 'name_Tata', 'name_Toyota',
       'name_Volkswagen', 'name_Volvo', 'fuel_Diesel', 'fuel_LPG',
       'fuel_Petrol', 'seller_type_Individual',
       'seller_type_Trustmark Dealer', 'transmission_Manual',
       'owner_Fourth & Above Owner', 'owner_Second Owner',
       'owner_Test Drive Car', 'owner_Third Owner']
    )).all(), f"Wrong order of columns after preprocessing: {df.columns.values}"

    return df


def predict_price(df: pd.DataFrame) -> np.array:
    prediction = MODEL_DATA["model"].predict(df)

    return np.e**prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_DATA
    MODEL_DATA = load_model_data()
    yield
    MODEL_DATA.clear()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    )
app = FastAPI(lifespan=lifespan)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return ...


@app.post("/predict_items")
def predict_items(file: UploadFile):
    dict_reader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    data = pd.DataFrame(list(dict_reader)).drop(columns=["", "torque"])
    
    for col in ["year", "km_driven"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    prepared_data = prepare_data(data)
    
    data["predicted_price"] = predict_price(prepared_data)
    data.to_csv('predicted.csv')

    response = FileResponse('predicted.csv', media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predicted.csv"
    return response


# if __name__ == "__name__":
    # uvicorn.run("app:app", host="0.0.0.0", port=8000)