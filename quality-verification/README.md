# Mealawe Quality Verification Pipeline

This repository contains the logic to run Vision Language Models (VLMs) over images of Indian thali meals, strictly outputting JSON based on the `[MEALAWE_THALI_QA_TEST_PROMPT_v0.1]` system prompt.

## Setup

1. **Install Dependencies:**
   Ensure you have Python 3.8+ installed. Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Keys:**
   Create a `.env` file in the root of the project (you can copy `.env.example`):
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   Alternatively, you can set them as environment variables directly.

   **Windows (Command Prompt):**
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```
   **Windows (PowerShell):**
   ```ps1
   $env:OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

1. **Place Testing Images:**
   Put any test images (`.jpg`, `.png`, `.jpeg`, `.webp`) in the `Testing images` directory.

2. **Run the Evaluation Pipeline:**
   Run the `evaluate.py` script.
   
   **For Google Gemini models:**
   ```bash
   python evaluate.py --img-dir "Testing images" --output-dir "results" --model-id "gemini-1.5-pro"
   ```

   **For OpenAI models:**
   ```bash
   python evaluate.py --img-dir "Testing images" --output-dir "results" --model-id "gpt-4o"
   ```

3. **Check Results:**
   The script will create a `results` folder and place JSON documents corresponding to each image evaluated, outputting strictly the validated schema specified by the prompt.

## File Overview
- `prompt_template.py`: Houses the core logic for the system prompt formatting.
- `evaluate.py`: Main executable script for reading images, calling the VLM, and processing responses.
- `requirements.txt`: Project dependencies (e.g., `requests`).
