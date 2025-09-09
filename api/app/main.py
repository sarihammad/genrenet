"""
FastAPI application for GTZAN genre classification.
"""
from fastapi import FastAPI, UploadFile, File, Query, Depends
from fastapi.responses import JSONResponse
import io
import os

from .schemas import PredictResponse
from .deps import get_model, get_label_map, get_device, get_config
from .infer import preprocess_audio, predict


app = FastAPI(
    title="GTZAN Genre Classification API",
    description="API for music genre classification using CNN on log-mel spectrograms",
    version="0.1.0"
)


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_genre(
    file: UploadFile = File(...),
    topk: int = Query(default=3, ge=1, le=10),
    model: torch.nn.Module = Depends(get_model),
    label_map: dict = Depends(get_label_map),
    device: torch.device = Depends(get_device),
    config: dict = Depends(get_config)
):
    """Predict genre from uploaded audio file."""
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Preprocess audio
        spec = preprocess_audio(
            audio_bytes,
            sample_rate=config['data']['sample_rate'],
            duration_sec=config['data']['duration_sec']
        )
        
        # Run inference
        predictions = predict(model, spec, label_map, device, topk)
        
        return PredictResponse(topk=predictions)
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to process audio: {str(e)}"}
        )


# Warmup on startup
@app.on_event("startup")
async def startup_event():
    """Warmup model on startup."""
    try:
        # Load dependencies to warm up
        model = get_model()
        label_map = get_label_map()
        device = get_device()
        config = get_config()
        
        # Create dummy input for warmup
        dummy_spec = torch.randn(1, 1, 128, 200).to(device)
        with torch.no_grad():
            _ = model(dummy_spec)
        
        print("Model warmed up successfully!")
        
    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8089))
    uvicorn.run(app, host="0.0.0.0", port=port)
