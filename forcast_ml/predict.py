from fastapi import FastAPI
from pydantic import BaseModel
from model import StockPricePredictor  # ← class ที่เราสร้างไว้
import uvicorn

app = FastAPI()

class PredictionRequest(BaseModel):
    ticker: str
    today: str  # Format: YYYY-MM-DD

@app.post("/predict")
def predict_stock_price(request: PredictionRequest):
    try:
        predictor = StockPricePredictor(request.ticker, request.today)
        predicted_price, plot_base64 = predictor.run()

        return {
            "predicted_price": float(predicted_price),  # ✅ fixed here
            "plot_base64": plot_base64
        }
    except Exception as e:
        return {"error": str(e)}