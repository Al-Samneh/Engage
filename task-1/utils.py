import os
import json
import datetime

def debug_log(step_name, data):
    """
    Writes input/output data to a JSON file in the debug folder.
    """
    os.makedirs("debug", exist_ok=True)
    
    filename = f"debug/{step_name}.json"
    
    try:
        with open(filename, "w", encoding='utf-8') as f:
            # Ensure non-serializable objects (like objects) are converted to str
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        print(f"Debug log saved: {filename}")
    except Exception as e:
        print(f"Could not write debug log for {step_name}: {e}")