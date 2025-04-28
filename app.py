from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Load model yang sudah disimpan
model = joblib.load('random_forest_regressor_best.pkl')

# Definisikan format input menggunakan Pydantic
class GDPFeatures(BaseModel):
    tahun_1970: float
    tahun_1971: float
    tahun_1972: float
    tahun_1973: float
    tahun_1974: float
    tahun_1975: float
    tahun_1976: float
    tahun_1977: float
    tahun_1978: float
    tahun_1979: float
    tahun_1980: float
    tahun_1981: float
    tahun_1982: float
    tahun_1983: float
    tahun_1984: float
    tahun_1985: float
    tahun_1986: float
    tahun_1987: float
    tahun_1988: float
    tahun_1989: float
    tahun_1990: float
    tahun_1991: float
    tahun_1992: float
    tahun_1993: float
    tahun_1994: float
    tahun_1995: float
    tahun_1996: float
    tahun_1997: float
    tahun_1998: float
    tahun_1999: float
    tahun_2000: float
    tahun_2001: float
    tahun_2002: float
    tahun_2003: float
    tahun_2004: float
    tahun_2005: float
    tahun_2006: float
    tahun_2007: float
    tahun_2008: float
    tahun_2009: float
    tahun_2010: float
    tahun_2011: float
    tahun_2012: float
    tahun_2013: float
    tahun_2014: float
    tahun_2015: float
    tahun_2016: float
    tahun_2017: float
    tahun_2018: float
    tahun_2019: float
    tahun_2020: float
    tahun_2021: float

@app.post("/predict")
def predict_gdp(features: GDPFeatures):
    # Ambil semua fitur secara dinamis
    input_data = np.array([
        features.tahun_1970, features.tahun_1971, features.tahun_1972,
        features.tahun_1973, features.tahun_1974, features.tahun_1975,
        features.tahun_1976, features.tahun_1977, features.tahun_1978,
        features.tahun_1979, features.tahun_1980, features.tahun_1981,
        features.tahun_1982, features.tahun_1983, features.tahun_1984,
        features.tahun_1985, features.tahun_1986, features.tahun_1987,
        features.tahun_1988, features.tahun_1989, features.tahun_1990,
        features.tahun_1991, features.tahun_1992, features.tahun_1993,
        features.tahun_1994, features.tahun_1995, features.tahun_1996,
        features.tahun_1997, features.tahun_1998, features.tahun_1999,
        features.tahun_2000, features.tahun_2001, features.tahun_2002,
        features.tahun_2003, features.tahun_2004, features.tahun_2005,
        features.tahun_2006, features.tahun_2007, features.tahun_2008,
        features.tahun_2009, features.tahun_2010, features.tahun_2011,
        features.tahun_2012, features.tahun_2013, features.tahun_2014,
        features.tahun_2015, features.tahun_2016, features.tahun_2017,
        features.tahun_2018, features.tahun_2019, features.tahun_2020,
        features.tahun_2021
    ]).reshape(1, -1)

    # Prediksi GDP tahun 2022
    prediction = model.predict(input_data)[0]

    return {"predicted_GDP_2022": prediction}