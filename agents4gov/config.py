import json
import os

CONFIG_PATH = os.getenv("AGENTS4GOV_CONFIG", "config_template.json")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
