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

# after testing options on june validation in modeling notebook, input size of 28 and max steps of 3000 were best
tft = TFT(h=1,input_size=28,max_steps=3000, futr_exog_list=futr, stat_exog_list=stat, scaler_type="robust", batch_size=256, val_check_steps=50,early_stop_patience_steps=5,random_seed =42, accelerator="cpu", enable_progress_bar=False)
nf = NeuralForecast(models=[tft], freq='D')#robust scaler was best for this case because daily sales are right skwed and in normal scale high days are thrown off

t = time.time()
#cv = nf.cross_validation(df=ydf, static_df=static_df, n_windows=5, step_size=1, val_size=60, refit=False)#first testing on some windows

# rolling one day ahead forecast
cv = nf.cross_validation(df=ydf, static_df=static_df, n_windows=61,step_size=1, val_size=60, refit = False)# refit false to train once then walking forward day by day not retraining every day
runtime_min = (time.time()- t)/60
n_params = sum(p.numel() for p in nf.models[0].parameters())
#print("cv done in", round(runtime_min,1), "min")
print("cv done in", round(runtime_min,1), "min, params", n_params)
