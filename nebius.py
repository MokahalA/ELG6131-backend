import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from google import genai
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not CLOUD_NAME or not CLOUDINARY_API_KEY or not CLOUDINARY_API_SECRET or not NEBIUS_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Cloudinary, Nebius, or Gemini API credentials are not set in environment variables")

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True  # Use HTTPS for all requests
)

# Configure Gemini API - Not needed when using Client API
# genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

origins = [
"https://www.e-hospital.ca",
"http://localhost:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_TOKENS = 1000
NEBIUS_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
GEMINI_MODEL_NAME = "gemini-2.0-flash" 

# Initialize Nebius client
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=NEBIUS_API_KEY
)

class AnalyzeRequest(BaseModel):
    image_url: str

async def analyze_image_with_gemini(image_url: str) -> dict:
    """Analyze lab requisition image using Gemini API."""
    try:
        import requests
        from google.genai import types
        
        # Download the image from URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch image from URL: {image_url}")
        
        # Prompt for lab requisition analysis
        prompt = """
            You are given this lab requisition form image to digitize, create a JSON with all of the fields, if a text field is empty then leave it blank. For the check boxes in Biochemistry, Hematology, Immunology, and Microbiology ID, Viral Hepatitis, PSA, and Vitamin D provide a 'YES' if the box is not empty. Do not include any quotation marks inside the field names. For any dates format them as: yyyymmdd. Provide only the JSON in the correct syntax. Follow the following template for the JSON.

            {
            "Name": "",
            "Address": "",
            "Clinician/Practitioner Number": "",
            "CPSO / Registration No.": "",
            "Clinician/Practitioner's Contact Number for Urgent Results": "",
            "Health Number": "",
            "Version": "",
            "Sex": "",
            "Service Date": "",
            "Date of Birth": "",
            "Province": "",
            "Other Provincial Registration Number": "",
            "Patient's Telephone Contact Number": "",
            "Patient's Last Name (as per OHIP Card)": "",
            "Patient's First Name (as per OHIP Card)": "",
            "Patient's Middle Name (as per OHIP Card)": "",
            "Patient's Address (including Postal Code)": "",
            "Check one": {
                "OHIP/Insured": "",
                "Third Party / Uninsured": "",
                "WSIB": ""
            },
            "Additional Clinical Information": "",
            "Copy to: Clinician/Practitioner": {
                "Last Name": "",
                "First Name": "",
                "Address": ""
            },
            "Biochemistry": {
                "Glucose": "",
                "Random": "",
                "Fasting": "",
                "HbA1C": "",
                "Creatinine (eGFR)": "",
                "Uric Acid": "",
                "Sodium": "",
                "Potassium": "",
                "ALT": "",
                "Alk. Phosphatase": "",
                "Bilirubin": "",
                "Albumin": "",
                "Lipid Assessment": "",
                "Albumin / Creatinine Ratio, Urine": "",
                "Urinalysis (Chemical)": "",
                "Neonatal Bilirubin": "",
                "Child's Age": "",
                "Clinician/Practitioner's tel. no.": "",
                "Patient's 24 hr telephone no.": "",
                "Therapeutic Drug Monitoring": {
                    "Name of Drug #1": "",
                    "Name of Drug #2": "",
                    "Time Collected #1": "",
                    "Time Collected #2": "",
                    "Time of Last Dose #1": "",
                    "Time of Last Dose #2": "",
                    "Time of Next Dose #1": "",
                    "Time of Next Dose #2": ""
                }
            },
            "Hematology": {
                "CBC": "",
                "Prothrombin Time (INR)": "",
                "Immunology": {
                    "Pregnancy Test (Urine)": "",
                    "Mononucleosis Screen": "",
                    "Rubella": "",
                    "Prenatal: ABO, RhD, Antibody Screen": "",
                    "Repeat Prenatal Antibodies": "",
                    "Microbiology ID & Sensitivities": {
                        "Cervical": "",
                        "Vaginal": "",
                        "Vaginal / Rectal - Group B Strep": "",
                        "Chlamydia (specify source)": "",
                        "GC (specify source)": "",
                        "Sputum": "",
                        "Throat": "",
                        "Wound (specify source)": "",
                        "Urine": "",
                        "Stool Culture": "",
                        "Stool Ova & Parasites": "",
                        "Other Swabs / Pus (specify source)": ""
                    }
                }
            },
            "Viral Hepatitis": {
                "Acute Hepatitis": "",
                "Chronic Hepatitis": "",
                "Immune Status / Previous Exposure": {
                    "Hepatitis A": "",
                    "Hepatitis B": "",
                    "Hepatitis C": ""
                }
            },
            "Prostate Specific Antigen (PSA)": {
                "Total PSA": "",
                "Free PSA": "",
                "Insured - Meets OHIP eligibility criteria": "",
                "Uninsured - Screening: Patient responsible for payment": ""
            },
            "Vitamin D (25-Hydroxy)": {
                "Insured - Meets OHIP eligibility criteria": "",
                "Uninsured - Patient responsible for payment": ""
            },
            "Other Tests": "",
            "I hereby certify the tests ordered are not for registered in or out patients of a hospital": "",
            "Specimen Collection": {
                "Time": "",
                "Date": ""
            },
            "Clinician/Practitioner Signature": "",
            "Date": ""
            }
        """
        
        # Create a client with the API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Create image part using Part.from_bytes
        image_part = types.Part.from_bytes(data=response.content, mime_type="image/jpeg")
        
        # Generate content using the Client API
        generation_response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,  # Use "gemini-2.0-flash" or your specific model name
            contents=[prompt, image_part]
        )
        
        output_text = generation_response.text.strip()
        if output_text.startswith("```json"):
            output_text = output_text[7:]
        if output_text.endswith("```"):
            output_text = output_text[:-3]
        
        # Return the response as JSON
        try:
            json_data = json.loads(output_text)
            return {"description": json_data, "status": "success"}
        except json.JSONDecodeError:
            # If parsing fails, return the raw text but with a warning
            return {"description": output_text, "status": "warning", "message": "Could not parse response as JSON"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image with Gemini: {str(e)}")

def analyze_image_with_nebius(image_url: str, prompt_text: str) -> dict:
    """Analyze image using Nebius API."""
    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert image analyst."}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ]
        response = client.chat.completions.create(
            model=NEBIUS_MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3
        )
        output_text = response.choices[0].message.content.strip()
        if output_text.startswith("```json"):
            output_text = output_text[7:]
        if output_text.endswith("```"):
            output_text = output_text[:-3]

        return {"description": json.loads(output_text), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image with Nebius: {str(e)}")

async def upload_file_to_cloudinary(file: UploadFile, folder: str):
    """Upload an image or convert PDF to JPG and upload to Cloudinary."""
    file_bytes = await file.read()
    upload_options = {
        "folder": folder,
        "resource_type": "auto", # let Cloudinary determine the type
    }
    if file.filename.lower().endswith(".pdf"):
      upload_options["format"] = "jpg" #convert to jpg
      upload_options["page"] = "1" #only first page
    response = cloudinary.uploader.upload(file_bytes, **upload_options)

    if not response.get("secure_url"):
        raise HTTPException(status_code=500, detail="Cloudinary upload failed.")
    return response["secure_url"]

@app.post("/upload-prescription/")
async def upload_prescription(file: UploadFile = File(...)):
    return {"message": "Prescription uploaded successfully", "url": await upload_file_to_cloudinary(file, "prescriptions")}

@app.get("/fetch-prescriptions/")
async def fetch_prescriptions():
    try:
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