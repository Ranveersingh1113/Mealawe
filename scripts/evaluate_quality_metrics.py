import os
import json
import glob
from collections import defaultdict

def evaluate_metrics(results_dir: str):
    """
    Reads all JSON files in the results directory, groups them by model, 
    and computes standard metrics (e.g. Failure rate, Missing Schema Fields).
    """
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' does not exist.")
        return

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print("No JSON results found. Run evaluate.py first with images.")
        return

    model_metrics = defaultdict(lambda: {
        "total_attempted": 0,
        "valid_json": 0,
        "missing_quantity": 0,
        "missing_quality": 0,
        "quantity_status_distribution": defaultdict(int),
        "quality_overall_distribution": defaultdict(int),
    })

    print(f"Analyzing {len(json_files)} result files...\n")
    for file_path in json_files:
        filename = os.path.basename(file_path)
        # We named files safe_model_id_imagefilename.json
        # Since images can be an arbitrary name, let's extract model from the file contents (meta tag).
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract meta
            meta = data.get("meta", {})
            model_id = meta.get("model_id", "Unknown")
            
            metrics = model_metrics[model_id]
            metrics["total_attempted"] += 1
            metrics["valid_json"] += 1
            
            quantity = data.get("quantity", {})
            if not quantity:
                metrics["missing_quantity"] += 1
            else:
                qs = quantity.get("overall_status", "missing")
                metrics["quantity_status_distribution"][qs] += 1
                
            quality = data.get("quality", {})
            if not quality:
                metrics["missing_quality"] += 1
            else:
                qo = quality.get("overall_rating", "missing")
                metrics["quality_overall_distribution"][qo] += 1
                
        except json.JSONDecodeError:
            print(f"Corrupt JSON file detected: {filename}")
            # Try to guess model from filename assuming standard pattern
            # Just add to an unknown failure bucket if we can't parse it
            model_metrics["Unknown (Parse Error)"]["total_attempted"] += 1

    # Print Report
    print("="*60)
    print(" " * 20 + "VLM METRICS REPORT")
    print("="*60)
    
    for model, m in model_metrics.items():
        if m["total_attempted"] == 0:
            continue
            
        success_rate = (m["valid_json"] / m["total_attempted"]) * 100
        
        print(f"Model ID: {model}")
        print(f"  - Total Evaluated : {m['total_attempted']}")
        print(f"  - JSON Parse Succ.: {success_rate:.1f}%")
        
        if m["valid_json"] > 0:
            print(f"  - Missing Quantity: {m['missing_quantity']} instances")
            print(f"  - Missing Quality : {m['missing_quality']} instances")
            print(f"  - Quantity Output Dist: {dict(m['quantity_status_distribution'])}")
            print(f"  - Quality Output Dist : {dict(m['quality_overall_distribution'])}")
        print("-" * 60)

if __name__ == "__main__":
    evaluate_metrics("results")
