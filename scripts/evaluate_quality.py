import os
import glob
import base64
import json
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv
from prompt_template import get_formatted_prompt

# Load environment variables from .env file if present
load_dotenv()

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def encode_image(image_path: str) -> str:
    """Read the image and encode it as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_vlm(image_path: str, model_id: str = "gpt-4o", medium: str = "api", api_url: str | None = None):
    """
    Sends the image and the system prompt to the Vision Language Model.
    Supports OpenAI-compatible APIs and Google Gemini API.
    """
    image_filename = os.path.basename(image_path)
    base64_image = encode_image(image_path)
    
    # 1. Format the text prompt using our template
    system_prompt = get_formatted_prompt(model_id=model_id, medium=medium, image_id=image_filename)

    # Extract format based on extension (simple approach)
    ext = image_filename.split(".")[-1].lower()
    mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"

    is_gemini = "gemini" in model_id.lower()

    if is_gemini:
        # Google Gemini API handling
        if not api_url:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
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
    else:
        # OpenAI-compatible API handling (including Ollama)
        if not api_url:
            # If it's a known OpenAI model, direct to OpenAI
            if model_id.startswith("gpt-"):
                api_url = "https://api.openai.com/v1/chat/completions"
            else:
                # Default to local Ollama instance for Qwen and other generic models
                api_url = "http://localhost:11434/v1/chat/completions"
                
        headers = {
            "Content-Type": "application/json"
        }
        if OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
            
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1500,
            "response_format": {"type": "json_object"} if model_id.startswith("gpt-") else None
        }
        if payload.get("response_format") is None:
            del payload["response_format"]

    print(f"Analyzing {image_filename} with {model_id}...")
    response = requests.post(api_url, headers=headers, json=payload)
    
    try:
        response.raise_for_status()
        response_json = response.json()
        
        # Extract the assistant's reply
        if is_gemini:
            content = response_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            content = response_json["choices"][0]["message"]["content"]
        
        # Ensure it's valid JSON
        parsed_result = json.loads(content)
        return parsed_result
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if response.text:
            print(f"Error details: {response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Raw response: {content}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run VLM Quality and Quantity Verification on Thali images.")
    parser.add_argument("--img-dir", type=str, default="Testing images", help="Directory containing images to test.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save JSON results.")
    parser.add_argument("--model-ids", type=str, default="gemini-2.5-flash,gemini-3.1-pro-preview,qwen2.5vl:7b-q4_K_M,qwen2.5:9b", help="Comma-separated List of Model IDs.")
    parser.add_argument("--api-url", type=str, default=None, help="Endpoint URL (Overrides automatic routing if set, so usually leave empty).")
    
    args = parser.parse_args()
    
    # Check if images directory exists
    if not os.path.exists(args.img_dir):
        print(f"Directory {args.img_dir} does not exist.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all common image formats
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))
        
    if not image_paths:
        print(f"No images found in {args.img_dir}. Insert test images to proceed.")
        return

    model_list = [m.strip() for m in args.model_ids.split(",")]

    for model_id in model_list:
        print(f"\n--- Starting evaluation for model: {model_id} ---")
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            
            # Since some models might fail or characters might be weird, we slugify the model id for filename
            safe_model_id = model_id.replace(":", "_").replace(".", "_")
            result_filename = f"{safe_model_id}_{os.path.splitext(filename)[0]}.json"
            result_path = os.path.join(args.output_dir, result_filename)

            # Skip if already done
            if os.path.exists(result_path):
                print(f"Skipping {filename} for {model_id}, already exists.")
                continue

            result_json = analyze_image_with_vlm(
                image_path=img_path,
                model_id=model_id,
                api_url=args.api_url
            )

            if result_json:
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2)
                print(f"Saved result: {result_path}")
            else:
                print(f"Failed to process {filename} with model {model_id}.")

if __name__ == "__main__":
    main()
