from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.prediction_pipeline import MobileData, PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline

app = FastAPI(title="Mobile Resale Value Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MobileInputData(BaseModel):
    brand: str
    model_name: str
    year_of_launch: int
    purchase_price_inr: float

class PredictionResponse(BaseModel):
    brand: str
    model_name: str
    year_of_launch: int
    purchase_price_inr: float
    predicted_resale_price_inr: float

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mobile Resale Value Predictor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: Segoe UI, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
  .wrapper { width: 100%; max-width: 560px; }
  .header { text-align: center; margin-bottom: 28px; }
  .header h1 { font-size: 26px; font-weight: 700; color: #ffffff; margin-bottom: 6px; }
  .header p { font-size: 14px; color: #94a3b8; }
  .badge { display: inline-block; background: rgba(109,40,217,0.3); color: #c4b5fd; font-size: 11px; font-weight: 600; padding: 4px 14px; border-radius: 20px; border: 1px solid rgba(109,40,217,0.5); margin-bottom: 14px; letter-spacing: 1px; text-transform: uppercase; }
  .card { background: #ffffff; border-radius: 20px; padding: 36px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
  .form-group { margin-bottom: 20px; }
  label { display: block; font-size: 12px; font-weight: 700; color: #6b7280; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  select, input { width: 100%; padding: 12px 14px; border: 1.5px solid #e5e7eb; border-radius: 10px; font-size: 14px; color: #111827; background: #f9fafb; outline: none; transition: all 0.2s; appearance: none; -webkit-appearance: none; cursor: pointer; }
  select { background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 14px center; padding-right: 36px; }
  select:focus, input:focus { border-color: #6d28d9; background: #ffffff; box-shadow: 0 0 0 3px rgba(109,40,217,0.1); }
  select:disabled { opacity: 0.5; cursor: not-allowed; }
  input[readonly] { background: #f3f0ff; border-color: #ddd6fe; color: #5b21b6; font-weight: 600; cursor: default; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .btn { width: 100%; padding: 14px; background: #6d28d9; color: white; border: none; border-radius: 10px; font-size: 15px; font-weight: 700; cursor: pointer; transition: all 0.2s; margin-top: 8px; }
  .btn:hover { background: #5b21b6; transform: translateY(-1px); }
  .btn:disabled { background: #a78bfa; cursor: not-allowed; transform: none; }
  .result { margin-top: 24px; background: linear-gradient(135deg, #6d28d9, #4f46e5); border-radius: 16px; padding: 28px; text-align: center; display: none; color: white; }
  .res-label { font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; opacity: 0.8; margin-bottom: 10px; }
  .res-price { font-size: 42px; font-weight: 800; margin-bottom: 12px; line-height: 1; }
  .res-row { display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.15); border-radius: 8px; padding: 8px 14px; margin-top: 8px; font-size: 13px; }
  .res-row span:first-child { opacity: 0.8; }
  .res-row span:last-child { font-weight: 700; }
  .dep-badge { display: inline-block; background: rgba(255,255,255,0.2); border-radius: 20px; padding: 4px 14px; font-size: 12px; font-weight: 600; margin-top: 12px; }
  .error { margin-top: 14px; background: #fef2f2; border: 1.5px solid #fecaca; border-radius: 8px; padding: 10px 14px; color: #b91c1c; font-size: 13px; display: none; }
  .loading { display: none; text-align: center; color: #6d28d9; font-size: 14px; font-weight: 600; margin-top: 14px; }
  .divider { border: none; border-top: 1px solid #f3f4f6; margin: 24px 0; }
  .train-btn { width: 100%; padding: 10px; background: white; color: #6d28d9; border: 1.5px solid #6d28d9; border-radius: 10px; font-size: 13px; font-weight: 600; cursor: pointer; }
  .train-btn:hover { background: #f5f3ff; }
  .train-msg { font-size: 13px; color: #059669; text-align: center; margin-top: 10px; font-weight: 600; display: none; }
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div class="badge">AI Powered</div>
    <h1>Mobile Resale Value Predictor</h1>
    <p>Select your mobile details to get the estimated resale price instantly</p>
  </div>
  <div class="card">
    <div class="form-group">
      <label>Brand</label>
      <select id="brand" onchange="onBrandChange()">
        <option value="">-- Select Brand --</option>
        <option>Apple</option>
        <option>Motorola</option>
        <option>Nokia</option>
        <option>OnePlus</option>
        <option>Oppo</option>
        <option>Realme</option>
        <option>Samsung</option>
        <option>Vivo</option>
        <option>Xiaomi</option>
      </select>
    </div>
    <div class="form-group">
      <label>Model Name</label>
      <select id="model_name" onchange="onModelChange()" disabled>
        <option value="">-- Select Model --</option>
      </select>
    </div>
    <div class="grid">
      <div class="form-group">
        <label>Year of Launch</label>
        <select id="year_of_launch" onchange="onYearChange()" disabled>
          <option value="">-- Select Year --</option>
        </select>
      </div>
      <div class="form-group">
        <label>Purchase Price (INR)</label>
        <input type="text" id="purchase_price_inr" placeholder="Auto-filled" readonly />
      </div>
    </div>
    <button class="btn" onclick="predict()">Get Resale Value</button>
    <div class="loading" id="loading">Predicting resale value...</div>
    <div class="error" id="error"></div>
    <div class="result" id="result">
      <div class="res-label">Estimated Resale Value</div>
      <div class="res-price" id="res_price"></div>
      <div class="res-row"><span>Original Purchase Price</span><span id="res_original"></span></div>
      <div class="res-row"><span>Depreciation Amount</span><span id="res_dep_amt"></span></div>
      <div class="dep-badge" id="res_dep_pct"></div>
    </div>
    <hr class="divider">
    <button class="train-btn" onclick="trainModel()">Retrain Model</button>
    <div class="train-msg" id="train_msg">Model retrained successfully!</div>
  </div>
</div>
<script>
const DB = {"Apple": {"iPhone 11": {"2018": 115648, "2021": 104411, "2020": 87768, "2024": 38741, "2019": 65625, "2023": 119328, "2022": 110714}, "iPhone 12": {"2018": 117145, "2021": 32086, "2022": 26249, "2023": 12505, "2024": 47398, "2020": 30775, "2019": 87245}, "iPhone 13": {"2020": 9662, "2023": 124330, "2024": 82312, "2019": 104499, "2022": 129211, "2018": 67663, "2021": 44050}, "iPhone 14": {"2019": 76777, "2020": 51480, "2021": 80887, "2022": 66531, "2024": 118678, "2018": 90912, "2023": 42798}}, "Motorola": {"Moto E40": {"2019": 88164, "2024": 28975, "2018": 16386, "2023": 47230, "2021": 71971, "2020": 102393, "2022": 23167}, "Moto Edge 20": {"2019": 20315, "2021": 121113, "2018": 128967, "2022": 127849, "2023": 107296, "2020": 123447, "2024": 74527}, "Moto G60": {"2018": 56606, "2021": 54623, "2022": 125520, "2023": 121311, "2019": 108118, "2020": 69822, "2024": 102491}, "Moto G71": {"2024": 90247, "2022": 123294, "2023": 9681, "2020": 112265, "2019": 82331, "2018": 16458, "2021": 20691}}, "Nokia": {"Nokia 5.4": {"2019": 8874, "2024": 98981, "2021": 45025, "2020": 123304, "2018": 38426, "2023": 68112, "2022": 109360}, "Nokia 6.2": {"2019": 108415, "2021": 102906, "2020": 11780, "2022": 111795, "2018": 108970, "2023": 95206, "2024": 26824}, "Nokia 7.2": {"2019": 115461, "2024": 20573, "2018": 116244, "2023": 58174, "2021": 109678, "2022": 76804, "2020": 44917}, "Nokia G20": {"2023": 58872, "2021": 127853, "2022": 91413, "2024": 118375, "2020": 62885, "2018": 99814, "2019": 21659}}, "OnePlus": {"OnePlus 10R": {"2022": 88004, "2021": 87600, "2020": 53918, "2023": 15433, "2018": 69647, "2019": 73810, "2024": 43050}, "OnePlus 8": {"2018": 105467, "2022": 73547, "2021": 109397, "2019": 16421, "2024": 53236, "2020": 43020, "2023": 69039}, "OnePlus 9": {"2020": 32834, "2021": 63845, "2024": 53414, "2022": 65085, "2023": 128866, "2018": 40199, "2019": 123765}, "OnePlus Nord": {"2021": 90799, "2019": 47643, "2018": 68074, "2023": 127673, "2022": 123674, "2020": 78514, "2024": 56497}}, "Oppo": {"Oppo A53": {"2022": 36743, "2024": 128029, "2021": 47905, "2023": 47483, "2020": 10855, "2018": 36481, "2019": 96604}, "Oppo F19": {"2019": 116705, "2022": 40371, "2020": 111432, "2021": 23409, "2018": 36596, "2023": 76622, "2024": 44476}, "Oppo F21": {"2019": 69987, "2023": 21558, "2022": 103238, "2018": 118555, "2020": 27696, "2024": 28640, "2021": 121629}, "Oppo Reno 6": {"2023": 69351, "2018": 25554, "2019": 48831, "2020": 118049, "2022": 44897, "2024": 82923, "2021": 73545}}, "Realme": {"Realme 8": {"2019": 23631, "2023": 26203, "2022": 58297, "2020": 100232, "2021": 124607, "2018": 125959, "2024": 120099}, "Realme 9": {"2020": 75881, "2021": 51118, "2023": 113815, "2018": 77738, "2019": 54966, "2024": 113394, "2022": 64608}, "Realme GT": {"2018": 64676, "2020": 97645, "2022": 60516, "2023": 64354, "2019": 11561, "2024": 93508, "2021": 57748}, "Realme Narzo 50": {"2018": 47673, "2021": 100451, "2023": 108331, "2020": 124344, "2024": 89178, "2019": 8745, "2022": 73709}}, "Samsung": {"Galaxy A52": {"2021": 110951, "2020": 16737, "2023": 39004, "2018": 99265, "2019": 17169, "2022": 80834, "2024": 30235}, "Galaxy S20": {"2020": 56289, "2023": 98096, "2019": 47961, "2018": 103326, "2022": 41614, "2021": 14440, "2024": 20450}, "Galaxy S21": {"2019": 91682, "2020": 104867, "2024": 77993, "2023": 120157, "2022": 72281, "2018": 91636, "2021": 111049}, "Galaxy S22": {"2024": 23789, "2023": 10860, "2022": 40229, "2021": 78963, "2019": 82820, "2020": 61831, "2018": 128078}}, "Vivo": {"Vivo V20": {"2023": 41492, "2024": 24692, "2019": 47491, "2021": 28140, "2018": 105337, "2020": 50449, "2022": 37423}, "Vivo V21": {"2023": 123023, "2022": 102662, "2018": 40768, "2024": 74091, "2021": 43903, "2019": 111945, "2020": 19039}, "Vivo X60": {"2018": 124491, "2023": 77360, "2021": 33445, "2020": 90532, "2022": 40173, "2024": 110488, "2019": 80387}, "Vivo Y20": {"2022": 42795, "2019": 115671, "2020": 123695, "2024": 7411, "2018": 51255, "2021": 128586, "2023": 76052}}, "Xiaomi": {"Mi 10": {"2023": 19511, "2024": 73162, "2022": 127989, "2018": 113582, "2020": 107811, "2021": 68097, "2019": 87446}, "Mi 11X": {"2018": 102891, "2023": 64246, "2024": 96557, "2021": 11658, "2022": 128049, "2020": 91824, "2019": 38981}, "Redmi Note 10": {"2023": 108771, "2021": 24296, "2020": 33952, "2022": 93482, "2018": 97750, "2024": 82557, "2019": 53663}, "Redmi Note 11": {"2019": 67939, "2024": 91578, "2020": 50279, "2018": 54072, "2022": 127579, "2023": 119940, "2021": 99901}}};

function onBrandChange() {
    const brand = document.getElementById('brand').value;
    const modelSel = document.getElementById('model_name');
    const yearSel = document.getElementById('year_of_launch');
    modelSel.innerHTML = '<option value="">-- Select Model --</option>';
    yearSel.innerHTML = '<option value="">-- Select Year --</option>';
    document.getElementById('purchase_price_inr').value = '';
    document.getElementById('result').style.display = 'none';
    yearSel.disabled = true;
    if (brand && DB[brand]) {
        Object.keys(DB[brand]).sort().forEach(m => { modelSel.innerHTML += '<option value="' + m + '">' + m + '</option>'; });
        modelSel.disabled = false;
    } else { modelSel.disabled = true; }
}

function onModelChange() {
    const brand = document.getElementById('brand').value;
    const model = document.getElementById('model_name').value;
    const yearSel = document.getElementById('year_of_launch');
    yearSel.innerHTML = '<option value="">-- Select Year --</option>';
    document.getElementById('purchase_price_inr').value = '';
    document.getElementById('result').style.display = 'none';
    if (brand && model && DB[brand][model]) {
        Object.keys(DB[brand][model]).sort().forEach(y => { yearSel.innerHTML += '<option value="' + y + '">' + y + '</option>'; });
        yearSel.disabled = false;
    } else { yearSel.disabled = true; }
}

function onYearChange() {
    const brand = document.getElementById('brand').value;
    const model = document.getElementById('model_name').value;
    const year = document.getElementById('year_of_launch').value;
    document.getElementById('result').style.display = 'none';
    if (brand && model && year && DB[brand][model][year]) {
        const price = DB[brand][model][year];
        document.getElementById('purchase_price_inr').value = price.toLocaleString('en-IN');
    } else { document.getElementById('purchase_price_inr').value = ''; }
}

async function predict() {
    const brand = document.getElementById('brand').value;
    const model_name = document.getElementById('model_name').value;
    const year_of_launch = parseInt(document.getElementById('year_of_launch').value);
    const priceStr = document.getElementById('purchase_price_inr').value.replace(/[,]/g, '');
    const purchase_price_inr = parseFloat(priceStr);
    document.getElementById('error').style.display = 'none';
    document.getElementById('result').style.display = 'none';
    if (!brand || !model_name || !year_of_launch || !purchase_price_inr) { showError('Please select Brand, Model and Year.'); return; }
    document.getElementById('loading').style.display = 'block';
    document.querySelector('.btn').disabled = true;
    try {
        const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ brand, model_name, year_of_launch, purchase_price_inr }) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');
        const predicted = data.predicted_resale_price_inr;
        const dep_amt = purchase_price_inr - predicted;
        const dep_pct = ((dep_amt / purchase_price_inr) * 100).toFixed(1);
        document.getElementById('res_price').innerText = 'Rs.' + Math.round(predicted).toLocaleString('en-IN');
        document.getElementById('res_original').innerText = 'Rs.' + purchase_price_inr.toLocaleString('en-IN');
        document.getElementById('res_dep_amt').innerText = 'Rs.' + Math.round(dep_amt).toLocaleString('en-IN');
        document.getElementById('res_dep_pct').innerText = 'Depreciated by ' + dep_pct + '%';
        document.getElementById('result').style.display = 'block';
    } catch(err) { showError(err.message); }
    finally { document.getElementById('loading').style.display = 'none'; document.querySelector('.btn').disabled = false; }
}

function showError(msg) { const el = document.getElementById('error'); el.innerText = msg; el.style.display = 'block'; }

async function trainModel() {
    document.querySelector('.train-btn').disabled = true;
    document.querySelector('.train-btn').innerText = 'Training...';
    document.getElementById('train_msg').style.display = 'none';
    try { await fetch('/train'); document.getElementById('train_msg').style.display = 'block'; }
    catch(e) { showError('Training failed: ' + e.message); }
    finally { document.querySelector('.train-btn').disabled = false; document.querySelector('.train-btn').innerText = 'Retrain Model'; }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: MobileInputData):
    try:
        mobile = MobileData(brand=data.brand, model_name=data.model_name, year_of_launch=data.year_of_launch, purchase_price_inr=data.purchase_price_inr)
        df = mobile.get_data_as_dataframe()
        pipeline = PredictionPipeline()
        predicted_price = pipeline.predict(df)
        return PredictionResponse(brand=data.brand, model_name=data.model_name, year_of_launch=data.year_of_launch, purchase_price_inr=data.purchase_price_inr, predicted_resale_price_inr=round(predicted_price, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train")
def train():
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return {"message": "Training pipeline completed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
