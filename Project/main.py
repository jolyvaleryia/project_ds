import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import model.pipeline


app = FastAPI()


class Form(BaseModel):
    session_id: Union[str, None] = None
    client_id: Union[str, None] = None
    visit_date: Union[str, None] = None
    visit_time: Union[str, None] = None
    visit_number: Union[int, None] = None
    utm_source: Union[str, None] = None
    utm_medium: Union[str, None] = None
    utm_campaign: Union[str, None] = None
    utm_adcontent: Union[str, None] = None
    utm_keyword: Union[str, None] = None
    device_category: Union[str, None] = None
    device_os: Union[str, None] = None
    device_brand: Union[str, None] = None
    device_model: Union[str, None] = None
    device_screen_resolution: Union[str, None] = None
    device_browser: Union[str, None] = None
    geo_country: Union[str, None] = None
    geo_city: Union[str, None] = None


class Prediction(BaseModel):
    session_id: str
    result: float


@app.get('/status')
def status():
    return "It's okay"


model = joblib.load('model/model_pipe_07312023_224525.pkl')


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'result': y[0]
    }

