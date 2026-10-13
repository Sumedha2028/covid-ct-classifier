
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.app.services.inference import load_model, predict_image


MAX_FILE_SIZE = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.classes = load_model()
    yield


app = FastAPI(
    title="CT Image Classifier API",
    description="Experimental CT image classification using MobileNetV3-Large.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://covid-ct-classifier.vercel.app"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "CT Image Classifier API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()

        with Image.open(BytesIO(content)) as image:
            result = predict_image(
                image,
                app.state.model,
                app.state.classes,
            )

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        )

    return {
        "filename": file.filename,
        **result,
        "notice": "Experimental output only. Not for medical diagnosis.",
    }
