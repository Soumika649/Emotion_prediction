
from contextlib import asynccontextmanager
import pickle
import re

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from pydantic import BaseModel, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "Artifacts/BiGRU_Modle.keras"
TOKENIZER_PATH = "Artifacts/tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 50

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


def preprocess_text(text: str) -> str:
    """Cleans raw text so it matches the format used while training."""
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"},
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilites: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


dl_model: dict = {}  # {"BiGRU": ..., "Tokenizer": ...}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")
    dl_model["BiGRU"] = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print("Model loaded successfully.")

    yield

    dl_model.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    """
    1. Cleans the input sentence.
    2. Converts the words into numeric tokens using the tokenizer.
    3. Pads the sequence to a uniform length.
    4. Runs prediction using the BiGRU model.
    5. Returns the top emotion plus the full probability breakdown.
    """
    bigru_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if bigru_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    cleaned_text = preprocess_text(text_input.text)

    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    probabilities = bigru_model.predict(padded_sequence)[0]

    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(prob) for prob, label in zip(probabilities, EMOTION_LABELS)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=EMOTION_LABELS[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilites=all_probabilities,
    )
