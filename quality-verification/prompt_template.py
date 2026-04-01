SYSTEM_PROMPT_TEMPLATE = """[MEALAWE_THALI_QA_TEST_PROMPT_v0.1]

(Experiment metadata — copy into JSON exactly)
model_id={{MODEL_ID}}            # e.g., "Qwen2-VL-2B-Instruct", "InternVL2-2B", "GPT-4o-mini"
medium={{MEDIUM}}                # "local" or "api"
image_id={{IMAGE_ID}}            # filename or unique id

INPUT:
- Full tray image: {{TRAY_IMAGE}}   # provide as attachment / image input / URL as your tool requires
- Quantity threshold for dal/sabzi/curry/rice: 0.75

TASK:
You are evaluating a single Indian thali (full tray image only).

1) QUANTITY VERIFICATION (visual estimate):
For each of {dal, sabzi, curry, rice} that is present in the tray, visually estimate the fill ratio (0.0–1.0):
- If estimated_fill_ratio >= 0.75 => status="verified"
- If estimated_fill_ratio < 0.75 => status="not_verified"
- If item/compartment is not visible or cannot be estimated reliably => status="unknown" and estimated_fill_ratio=null

Overall quantity status:
- "not_verified" if ANY present item among {dal,sabzi,curry,rice} is not_verified
- "verified" if ALL present items among {dal,sabzi,curry,rice} are verified
- otherwise "unknown"

2) QUALITY ASSESSMENT (Good / Acceptable / Poor) for ALL visible elements:
Evaluate quality for every element you can see, including:
- dal, sabzi, curry, rice, roti, salad, sweet (if present), and tray_cleanliness.

QUALITY GUIDELINES (use visual cues):

DAL:
- Good: consistent texture, appetizing color, fresh cooked look, not watery-separated, no burn/contamination.
- Acceptable: slight thin/thick, minor oil separation, minor mess, still edible.
- Poor: very watery-separated/curdled, burnt/blackened, stale-looking, contamination/foreign object, very messy.

SABZI:
- Good: vibrant color, properly cooked, not burnt, not overly oily, looks fresh.
- Acceptable: slightly dull/uneven, mild dryness or mild under/overcook, still fine.
- Poor: burnt/blackened, extremely oily, stale/dry/shriveled, raw-looking, suspicious discoloration, contamination.

CURRY:
- Good: consistent gravy, good color, not watery/oily separated, fresh look.
- Acceptable: minor oil separation, slightly watery/thick, minor spillage.
- Poor: heavy separation/curdling, burnt edges, watery+solids separated, contamination.

RICE:
- Good: fluffy/separate grains, clean appearance, no foreign matter.
- Acceptable: slightly clumpy or slightly dry, still edible.
- Poor: very clumpy/gummy, stale/dry, discolored, dirty/foreign particles, major mess.

ROTI:
- Good: evenly cooked, normal brown spots, fresh/soft look, not badly torn.
- Acceptable: slightly dry, minor tear, slightly overcooked spots.
- Poor: burnt large areas, very dry/hard, badly torn/crumbled, stale-looking, contamination.

SALAD:
- Good: fresh/crisp, clean cuts, no browning/slime, hygienic.
- Acceptable: slight wilting or mild browning edges.
- Poor: soggy/slimy, heavy browning, stale/unclean, contamination.

SWEET (if present):
- Good: intact, appealing color, not melted/messy, fresh look.
- Acceptable: minor deformation, slightly syrupy/messy.
- Poor: heavily melted/smeared, odd discoloration, contamination, stale look.

TRAY_CLEANLINESS / PRESENTATION:
- Good: clean tray/compartments, neat, minimal spills.
- Acceptable: small spills/smudges, minor mess.
- Poor: major spills across compartments, dirty edges, suspected hair/foreign object, unhygienic.

OUTPUT REQUIREMENTS:
Return ONLY valid JSON (no markdown, no extra text) with exactly this structure:

{
  "meta": {
    "model_id": "{{MODEL_ID}}",
    "medium": "{{MEDIUM}}",
    "image_id": "{{IMAGE_ID}}"
  },
  "quantity": {
    "threshold": 0.75,
    "items": [
      {"item":"dal","estimated_fill_ratio":null,"status":"unknown","reason":""},
      {"item":"sabzi","estimated_fill_ratio":null,"status":"unknown","reason":""},
      {"item":"curry","estimated_fill_ratio":null,"status":"unknown","reason":""},
      {"item":"rice","estimated_fill_ratio":null,"status":"unknown","reason":""}
    ],
    "overall_status": "unknown"
  },
  "quality": {
    "items": [
      {"item":"dal","present":false,"rating":"unknown","confidence":0.0,"issues":[],"evidence":""},
      {"item":"sabzi","present":false,"rating":"unknown","confidence":0.0,"issues":[],"evidence":""},
      {"item":"curry","present":false,"rating":"unknown","confidence":0.0,"issues":[],"evidence":""},
      {"item":"rice","present":false,"rating":"unknown","confidence":0.0,"issues":[],"evidence":""},
      {"item":"roti","present":false,"rating":"unknown","confidence":0.0,"issues":[],"evidence":""},
      {"item":"salad","present":false,"rating":"unknown","confidence":0.0,"issues":[],"evidence":""},
      {"item":"sweet","present":false,"rating":"unknown","confidence":0.0,"issues":[],"evidence":""},
      {"item":"tray_cleanliness","present":true,"rating":"unknown","confidence":0.0,"issues":[],"evidence":""}
    ],
    "overall_rating": "unknown",
    "critical_flags": []
  },
  "notes": ""
}

FILLING RULES:
- Set present=true only if clearly visible.
- If dal/sabzi/curry/rice present, estimate fill ratio if possible; else unknown.
- confidence must be 0.0-1.0.
- Keep reason/evidence concise (1 sentence each).
- Add critical_flags for contamination/foreign object, severe spillage, or hygiene issues.
"""

def get_formatted_prompt(model_id: str, medium: str, image_id: str) -> str:
    """
    Replaces the placeholders in the prompt with actual metadata.
    Note: {{TRAY_IMAGE}} is handled via API attachments usually, but we keep the placeholder text or replace it.
    """
    prompt = SYSTEM_PROMPT_TEMPLATE.replace("{{MODEL_ID}}", model_id)
    prompt = prompt.replace("{{MEDIUM}}", medium)
    prompt = prompt.replace("{{IMAGE_ID}}", image_id)
    prompt = prompt.replace("{{TRAY_IMAGE}}", "Attached in API request")
    return prompt
