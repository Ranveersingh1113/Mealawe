import base64
import json
import os
import requests
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from app.schemas import QualityStageResponse

# Define the pipeline directory and load its specific .env file
PIPELINE_DIR = Path(__file__).resolve().parent.parent / "quality-verification"
load_dotenv(PIPELINE_DIR / ".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Import the exact same prompt template from the verification pipeline
sys.path.append(str(PIPELINE_DIR))
from prompt_template import get_formatted_prompt

def call_gemini(model_id: str, system_prompt: str, base64_image: str) -> dict:
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    # Strip markdown block wrappers if present
    content = content.replace("```json", "").replace("```", "").strip()
    
    return json.loads(content)
def run_quality_stage(image_bytes: bytes, compartments: List[Dict[str, Any]]) -> QualityStageResponse:
    """
    Executes the VLM integration using Gemini 3.1 Pro Preview with fallback to 2.5 Flash.
    Connects CV-detected compartments with whole-pipeline quantity & quality prompts.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    try:
        # Try primary model
        system_prompt = get_formatted_prompt(model_id="gemini-3.1-pro-preview", medium="api", image_id="api_request")
        vlm_result = call_gemini("gemini-3.1-pro-preview", system_prompt, base64_image)
    except Exception as e:
        print(f"Fallback triggered. Gemini-3.1-pro-preview failed: {e}")
        # Fallback to flash
        try:
            system_prompt = get_formatted_prompt(model_id="gemini-2.5-flash", medium="api", image_id="api_request")
            vlm_result = call_gemini("gemini-2.5-flash", system_prompt, base64_image)
        except Exception as fallback_e:
            return QualityStageResponse(
                status="VLM_ERROR",
                message=f"Both primary and fallback models failed. Error: {fallback_e}",
                compartment_candidates=compartments
            )

    # Determine status from VLM output
    if isinstance(vlm_result, list) and len(vlm_result) > 0:
        vlm_result = vlm_result[0]

    vlm_quality = vlm_result.get("quality", {})
    overall_quality = vlm_quality.get("overall_rating", "unknown")
    # We embed the VLM evaluation into the response mapping
    return QualityStageResponse(
        status=overall_quality.upper(),
        message="Successfully evaluated Thali quality and quantity using Vision Language Models.",
        compartment_candidates=[vlm_result] 
    )

