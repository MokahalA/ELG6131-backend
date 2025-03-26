import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils import (
    upload_file_to_cloudinary, 
    analyze_image_with_nebius, 
    analyze_image_with_gemini,
    AnalyzeRequest
)

app = FastAPI()

# CORS configuration - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware to ensure CORS headers are always present
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Handle OPTIONS requests explicitly
@app.options("/{path:path}")
async def options_handler(path: str):
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.post("/upload-prescription/")
async def upload_prescription(file: UploadFile = File(...)):
    return {"message": "Prescription uploaded successfully", "url": await upload_file_to_cloudinary(file, "prescriptions")}

@app.get("/fetch-prescriptions/")
async def fetch_prescriptions():
    try:
        import cloudinary.api
        
        response = cloudinary.api.resources(type="upload", resource_type="image", prefix="prescriptions/")
        return {"images": [item["secure_url"] for item in response.get("resources", [])]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prescriptions: {str(e)}")

@app.post("/analyze-prescription/")
async def analyze_prescription(request: AnalyzeRequest):
    prompt_text = "For this prescription document image, provide a JSON with fields: medications (name, dosage, frequency), instructions (summary of the prescription). Provide only the JSON and only these fields."
    return analyze_image_with_nebius(request.image_url, prompt_text)

@app.post("/upload-lab-requisition/")
async def upload_lab_requisition(file: UploadFile = File(...)):
    return {"message": "Lab requisition uploaded successfully", "url": await upload_file_to_cloudinary(file, "lab-requisitions")}

@app.get("/fetch-lab-requisitions/")
async def fetch_lab_requisitions():
    try:
        import cloudinary.api
        
        response = cloudinary.api.resources(type="upload", resource_type="image", prefix="lab-requisitions/")
        return {"images": [item["secure_url"] for item in response.get("resources", [])]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching lab requisitions: {str(e)}")

@app.post("/analyze-lab-requisition/")
async def analyze_lab_requisition(request: AnalyzeRequest):
    # Use Gemini API for lab requisition analysis
    return await analyze_image_with_gemini(request.image_url)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)