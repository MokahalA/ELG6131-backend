import os
import base64
import io
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
import uvicorn
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

if not CLOUD_NAME or not CLOUDINARY_API_KEY or not CLOUDINARY_API_SECRET or not NEBIUS_API_KEY:
    raise ValueError("Cloudinary or Nebius API credentials are not set in environment variables")

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True  # Use HTTPS for all requests
)

app = FastAPI()

# Configuration
MAX_TOKENS = 300
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# Initialize Nebius client
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=NEBIUS_API_KEY
)

class AnalyzeImageRequest(BaseModel):
    image_url: str

def encode_image(image_bytes: bytes) -> str:
    """Encodes image bytes into a Base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")

async def analyze_image(image_url: str) -> dict:
    """Analyze image using Nebius API."""
    try:
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an expert image analyst."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "For this document image, give me a JSON with the fields: medications (name, dosage, frequency), instructions (summary of the prescription). Provide only the JSON and only these fields."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3
        )
        output_text = response.choices[0].message.content.strip()
        if output_text.startswith("```json"):
            output_text = output_text[7:]
        if output_text.endswith("```"):
            output_text = output_text[:-3]
        try:
            parsed_json = json.loads(output_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Model returned invalid JSON.")
        return {"description": parsed_json, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to Cloudinary."""
    file_bytes = await file.read()
    # Upload image to Cloudinary using the SDK with the 'prescriptions' folder
    response = cloudinary.uploader.upload(file_bytes, resource_type="image", folder="prescriptions")

    if response.get("secure_url") is None:
        raise HTTPException(status_code=500, detail="Cloudinary upload failed.")

    image_url = response["secure_url"]
    return {"message": "Image uploaded successfully", "url": image_url}

@app.get("/fetch-images/")
async def fetch_images():
    """Fetch all images stored in the 'prescriptions' folder in Cloudinary."""
    try:
        response = cloudinary.api.resources(
            type="upload",
            resource_type="image",
            prefix="prescriptions/"
        )
        image_urls = [item["secure_url"] for item in response.get("resources", [])]
        return {"images": image_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching images: {str(e)}")

@app.post("/analyze-image/")
async def analyze_image_endpoint(request: AnalyzeImageRequest):
    """Analyze image using Nebius API."""
    image_url = request.image_url
    return await analyze_image(image_url)

# Not needed for deployment
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
