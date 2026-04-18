"""
api.py
FastAPI backend — /predict endpoint.
Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

from inference import predict, load_ensemble, load_knowledge_base

app = FastAPI(title="Plant Disease API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
def read_root():
    return RedirectResponse(url="/app/")

# Pre-load model at startup
@app.on_event("startup")
async def startup():
    print("Loading ensemble model...")
    app.state.model_loaded = True
    print("Model ready.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def get_classes():
    import json
    from pathlib import Path
    classes_path = Path(r"C:\Project\Emseble\checkpoints\classes.json")
    if not classes_path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet")
    with open(classes_path) as f:
        return {"classes": json.load(f)}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    # Validate image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save to temp file (inference.py needs a path)
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot open image file")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, "JPEG")
        tmp_path = tmp.name

    try:
        result = predict(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
