from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict

app = FastAPI()

# allow next.js app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI model
model = joblib.load("price_model.pkl")

# entered data by user
class EstateInput(BaseModel):
    wilaya: str
    baladia: str
    type: str
    surface: float
    rooms: int
    floor: int
    estateCarac: str
    zone: str

# mapping
wilaya_map = {
    "04 - أم البواقي": 4,
}

baladia_map = {
    "أم البواقي": 2,
    "عين البيضاء": 1,
    "عين مليلة": 3,
    "عين فكرون": 4,
}

type_map = {
    "شقة": 1,
    "منزل / فيلا": 2,
    "محل": 3,
    "قطعة أرض": 4,
}

zone_map = {
    "وسط المدينة": 1,
    "قريب من الوسط": 2,
    "أطراف المدينة": 3,
    "خارج المدينة": 4,
}

carac_map = {
    "فاخر": 1,
    "محسن ( فيه تحسينات و إضافات )": 2,
    "عادي": 3,
    "قديم / بناء غير منتهي": 4,
}

@app.get("/")
def root():
    return {"message": "Estate Price API is running", "status": "healthy"}

@app.post("/predict")
def predict_price(estate: EstateInput):
    try:
        wilaya_num = wilaya_map.get(estate.wilaya)
        baladia_num = baladia_map.get(estate.baladia)
        type_num = type_map.get(estate.type)
        zone_num = zone_map.get(estate.zone)
        carac_num = carac_map.get(estate.estateCarac)

        if None in [wilaya_num, baladia_num, type_num, zone_num, carac_num]:
            raise HTTPException(status_code=400, detail="invalid value in an input")

        year_of_sell = 2026

        features = pd.DataFrame([[
            wilaya_num,
            baladia_num,
            type_num,
            estate.surface,
            estate.rooms,
            estate.floor,
            year_of_sell,
            carac_num,
            zone_num
        ]], columns=[
            "wilaya", "baladia", "type", "surface", "rooms",
            "floor", "YearOfSell", "estateCarac", "zone"
        ])

        prediction = model.predict(features)[0]
        
        predicted_price = round(prediction, -2)

        return {
            "estimated_price_da": predicted_price,
            "estimated_price_formatted": f"{predicted_price:,} DA",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f" prediction error : {str(e)}")
