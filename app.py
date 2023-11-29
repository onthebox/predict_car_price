import codecs
import csv
import logging
import pickle
import re
from contextlib import asynccontextmanager
from typing import Annotated, List, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


class Item(BaseModel):
    """Define Item object for request data validation.
    """
    name: str
    year: int
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
    """Define list of Items objects.
    
    Currently unused.
    """
    objects: List[Item]


def load_model_data():
    with open("model_data.pickle", "rb") as f:
        model_data = pickle.load(f)

    return model_data


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe by adding new features and modifying existing.
    
    Args:
        - df -- datadrame to modify

    Returns:
        df -- modified dataframe
    """
    def get_value(x: str) -> Union[float, None]:
        """Get int- or float-like value from string which contains different characters.
        
        Args:
            - x -- string to extract value from.

        Returns:
            Either extarcted float or None if no values occured in string.
        """
        if not pd.notnull(x):
            return None

        value = re.search(r'\d+.\d+|\d+', x)
        if not value:
            return None

        return float(value.group())

    # Get float values from object columns
    for col in ["mileage", "engine", "max_power"]:
        df[col] = df[col].apply(get_value).astype(float)

    # Get car brand from 'name' column
    df["name"] = df["name"].apply(lambda x: x.split(' ')[0])

    # Split data to numerical and categorical
    categorical = df[["name", "fuel", "seller_type", "transmission", "owner"]]
    numerical = df.drop(columns=categorical.columns)

    # Preprocess numerical data; add new features
    numerical["power_per_volume"] =  numerical["max_power"] /   numerical["engine"]
    numerical["year_square"] = numerical["year"]**2
    numerical["engine_log"] = np.log(numerical["engine"])
    numerical["km_driven_log"] = np.log(numerical["km_driven"])

    # Normalize numerical data with StandardScaler
    numerical = pd.DataFrame(MODEL_DATA["StSc"].transform(numerical), columns=numerical.columns)

    # Encode categorical data with OneHotEncoder
    categorical = pd.DataFrame(MODEL_DATA["OHE"].transform(categorical).toarray(),
                               columns=MODEL_DATA["OHE"].get_feature_names_out())
    
    # Merge back to one DataFrame
    df = pd.concat([numerical, categorical], axis=1)

    # Assertion on the right columns order
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
    """Predict price based on preprocessed data with loaded Ridge regression model.

    Args:
        - df -- DataFrame with preprocessed data

    Returns:
        - np.array -- predicted car price 
    """
    prediction = MODEL_DATA["model"].predict(df)

    # Notice the exp function, as the model returns log(price)
    return np.e**prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Ridge Regression model and needed data from pickle file on startup.
    """
    global MODEL_DATA
    MODEL_DATA = load_model_data()
    yield
    MODEL_DATA.clear()


# Set logging config
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    )

app = FastAPI(
    lifespan=lifespan,
    title="Predict the price of a car",
    version="1.0",
    contact={
        "name": "Viktor Tikhomirov",
        "url": "https://github.com/onthebox",
        "email": "tikhomirov65@gmail.com"
    }
    )


@app.post("/predict_item", summary="Predict the price of a car")
def predict_item(item: Item) -> float:
    """Predict the price of a car based on input car information.

    Args:
        - item -- car data

    Returns:
        float -- predicted car price
    """
    # Dump Item first to dict and than to DataFrame
    item_dict = item.model_dump(exclude="torque")
    item_df = pd.DataFrame(item_dict, index=[0])
    logging.info(f"Successfully loaded car data to DataFrame")

    # Prepare car data and make prediction
    # Notice that predict_data returns np.array
    prepared_item_df = prepare_data(item_df)
    predicted_price = predict_price(prepared_item_df)
    logging.info(f"Successfully predicted price: {predicted_price}")

    return predicted_price[0]


@app.post("/predict_items", summary="Predict car prices")
def predict_items(file: Annotated[UploadFile, File(description="A file read as UploadFile")]) -> FileResponse:
    """Predict car prices based on cars data from uploaded .csv file.

    Args:
        file -- csv file with car data

    Returns:
        response -- response with csv file  supplemented by car prices.
    """

    # Check if file type is .csv
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid document type")

    # Read file data and load it to DataFrame
    dict_reader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    data = pd.DataFrame(list(dict_reader)).drop(columns=["torque"])
    logging.info(f"Loaded data to DataFrame. {data.shape} - shape; {data.columns} - columns.")
    
    # Cast required columns to numeric 
    for col in ["year", "km_driven"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Prepare data for prediction and predict price
    # Predicted price save as new column of original file
    prepared_data = prepare_data(data)
    logging.info(f"Prepared data for making predictions.")
    data["predicted_price"] = predict_price(prepared_data)
    logging.info(f"Successfully made predictions. {data['predicted_price'].values}")

    # Save new file
    data.to_csv('predicted.csv')
    logging.info(f"Saved original data with predictions.")

    # Prepare filename for file that will be downloaded
    # Form the response
    filename = file.filename[:-4] + '_predicted.csv'
    response = FileResponse(path='predicted.csv', media_type='text/csv', filename=filename)

    return response


# if __name__ == "__name__":
    # uvicorn.run("app:app", host="0.0.0.0", port=8000)