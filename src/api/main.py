"""FastAPI."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
app = FastAPI(title="House Price API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    n_samples: int = Field(default=1000, ge=100)

@app.get("/health")
async def health(): return {"status": "healthy"}

@app.post("/predict")
async def predict(req: PredictRequest):
    from src.price_model import generate_synthetic_data, train_and_evaluate
    df = generate_synthetic_data(n_samples=req.n_samples)
    results = train_and_evaluate(df)
    return {"n_samples": len(df), "results": results}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8012)
