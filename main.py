import os
import google.generativeai as genai
import medpalm
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# Load API key and Med-PaLM credentials
load_dotenv()

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize FastAPI
app = FastAPI(title="Google Generative AI API", version="1.1")

# Cache available models
try:
    available_models = {model.name: model for model in genai.list_models()}
except Exception as e:
    available_models = {}
    print(f"Error fetching model list: {e}")

# Med-PaLM Authentication Setup
MEDPALM_CREDENTIALS = os.getenv("MEDPALM_CREDENTIALS_PATH")  # Path to JSON file

# Check if Med-PaLM credentials exist
if MEDPALM_CREDENTIALS and os.path.exists(MEDPALM_CREDENTIALS):
    try:
        medpalm_client = medpalm.Client(
            email=os.getenv("MEDPALM_SERVICE_ACCOUNT_EMAIL"),
            key_file=MEDPALM_CREDENTIALS,
        )
        is_medpalm_available = True
    except Exception as e:
        print(f"Error initializing Med-PaLM: {e}")
        is_medpalm_available = False
else:
    is_medpalm_available = False


# 📌 Health Check Endpoint
@app.get("/", tags=["Health Check"])
def home():
    """Check if API is running and list available models."""
    return {
        "message": "Google Generative AI API is running!",
        "available_models": list(available_models.keys()),
        "medpalm_available": is_medpalm_available,
    }


# 📌 Get List of Available Models
@app.get("/models/", tags=["Models"])
def list_models():
    """Returns the list of available models."""
    return {"available_models": list(available_models.keys())}


# 📌 Request Model Input Schema
class ModelRequest(BaseModel):
    query: str
    model_name: str = "models/gemini-2.0-flash"


# 📌 General Model Generation Endpoint
@app.post("/generate/", tags=["General Models"])
async def generate_response(request: ModelRequest):
    """
    Generate text using a selected model.
    Default model: `models/gemini-2.0-flash`
    """
    model_name = request.model_name

    if model_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available models: {list(available_models.keys())}",
        )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(request.query)
        return {"model": model_name, "response": response.text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


# 📌 Med-PaLM 2 Specific Endpoint
@app.post("/medpalm/", tags=["Med-PaLM 2"])
async def medpalm_response(
    query: str = Query(..., description="Medical query for Med-PaLM 2")
):
    """
    Generate medical-specific responses using Med-PaLM 2.
    """
    if not is_medpalm_available:
        raise HTTPException(
            status_code=404,
            detail="Med-PaLM 2 model is not available. Ensure you have set up your credentials correctly.",
        )

    try:
        response = medpalm_client.ask_medical_question(query)
        return {"model": "Med-PaLM 2", "response": response.text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )
