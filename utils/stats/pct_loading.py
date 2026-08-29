import json
import os


def _load_pct(results_dir, model_name, condition):
    if condition == "base":
        candidates = [os.path.join(results_dir, "base", f"{model_name}.json")]
    else:
        candidates = [
            os.path.join(results_dir, "finetuned", f"{model_name}_{condition}.json")
        ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None