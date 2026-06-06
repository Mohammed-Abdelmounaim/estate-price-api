from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# محاولة تحميل النموذج مع طباعة أخطاء مفصلة
print("=== بدء تحميل التطبيق ===")
print("المجلد الحالي:", os.getcwd())
print("الملفات الموجودة:", os.listdir())

try:
    model = joblib.load("price_model.pkl")
    print("✅ تم تحميل النموذج بنجاح")
    model_loaded = True
except Exception as e:
    print(f"❌ فشل تحميل النموذج: {str(e)}")
    model_loaded = False
    model = None

class EstateInput(BaseModel):
    wilaya: str
    baladia: str
    type: str
    surface: float
    rooms: int
    floor: int
    estateCarac: str
    zone: str

# قواميس الترجمة
wilaya_map = {"04 - أم البواقي": 4}
baladia_map = {"أم البواقي": 2, "عين البيضاء": 1, "عين مليلة": 3, "عين فكرون": 4}
type_map = {"شقة": 1, "منزل / فيلا": 2, "محل": 3, "قطعة أرض": 4}
zone_map = {"وسط المدينة": 1, "قريب من الوسط": 2, "أطراف المدينة": 3, "خارج المدينة": 4}
carac_map = {"فاخر": 1, "محسن ( فيه تحسينات و إضافات )": 2, "عادي": 3, "قديم / بناء غير منتهي": 4}

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model_loaded, "message": "API is running"}

@app.get("/test")
def test():
    return {"message": "Test endpoint works"}

@app.post("/predict")
def predict_price(estate: EstateInput):
    print("=== استلام طلب تنبؤ ===")
    print("البيانات المستلمة:", estate.dict())
    
    if not model_loaded:
        print("❌ النموذج غير محمل")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # التحقق من صحة البيانات
        wilaya_num = wilaya_map.get(estate.wilaya)
        baladia_num = baladia_map.get(estate.baladia)
        type_num = type_map.get(estate.type)
        zone_num = zone_map.get(estate.zone)
        carac_num = carac_map.get(estate.estateCarac)
        
        print(f"قيم مترجمة: wilaya={wilaya_num}, baladia={baladia_num}, type={type_num}, zone={zone_num}, carac={carac_num}")
        
        if None in [wilaya_num, baladia_num, type_num, zone_num, carac_num]:
            print("❌ قيم غير صالحة")
            raise HTTPException(status_code=400, detail="قيمة غير صالحة في أحد الحقول")
        
        year_of_sell = 2026
        
        # إنشاء DataFrame
        features = pd.DataFrame([[
            wilaya_num, baladia_num, type_num, estate.surface, 
            estate.rooms, estate.floor, year_of_sell, carac_num, zone_num
        ]], columns=[
            "wilaya", "baladia", "type", "surface", "rooms",
            "floor ", "YearOfSell", "estateCarac", "zone"
        ])
        
        print("البيانات للتنبؤ:", features.to_dict())
        
        # التنبؤ
        prediction = model.predict(features)[0]
        predicted_price = round(prediction, -2)
        
        print(f"✅ السعر المتوقع: {predicted_price}")
        
        return {
            "estimated_price_dzd": int(predicted_price),
            "estimated_price_formatted": f"{int(predicted_price):,} DZD",
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ خطأ في التنبؤ: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"prediction error: {str(e)}")
