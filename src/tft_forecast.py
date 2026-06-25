import warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
np.random.seed(42)

# tft is very slower than the other 3 models and I designed it to run here and the modeling notebook calling this script

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLEAN = ROOT / 'data' / "processed" / "rossmann_clean.csv"
OUT = ROOT / 'results'
#OUT.mkdir()
OUT.mkdir(exist_ok= True)

df = pd.read_csv(CLEAN,parse_dates=['Date'], dtype={"StateHoliday" : str})
#category cols to codes for neuralforcast that needs numbers
df["StateHoliday_code"] = df["StateHoliday"].astype('category').cat.codes
df["StoreType_code"] = df["StoreType"].astype('category').cat.codes
df["Assortment_code"] = df["Assortment"].astype('category').cat.codes
df["DayOfWeek"]=df["DayOfWeek"].astype(int)
df["month"] = df["Date"].dt.month

futr = ["Promo", "SchoolHoliday", 'Open', "StateHoliday_code","month", "DayOfWeek"] #things we already know for each future day

# things that dont change for a store
stat = ["StoreType_code","Assortment_code","CompetitionDistance","Promo2"]
long=df.rename(columns={"Store" : "unique_id", "Date" : 'ds', "Sales" : "y"})
ydf = long[["unique_id",'ds','y'] +futr]
#ydf = long[["unique_id",'ds','y']]
static_df = long.groupby("unique_id")[stat].first().reset_index()
