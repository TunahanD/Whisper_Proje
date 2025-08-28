import re
import json
import os
import shutil

patterns=[
    {
        "intent":"camera_open",
        "pattern":re.compile(r"^(.*?) kamera görüntüsünü aç\.?$",re.IGNORECASE),
        "slots":["camera_name"]
    },
    {
        "intent":"camera_rotate",
        "pattern":re.compile(r"^(.*?) kamerasını (.+?) çevir\.?",re.IGNORECASE),
        "slots":["camera_name","preset_order"]
    },
    {
        "intent":"dark_mode",
        "pattern":re.compile(r"^ dark mode aç\.?$",re.IGNORECASE),
        "slots":[]
    },
    {
        "intent":"light_mode",
        "pattern":re.compile(r"^ light mode aç\.?$",re.IGNORECASE),
        "slots":[]
    },
]

def parse_intent(user_input:str):
    user_input=user_input.strip()
    for p in patterns:
        match=p["pattern"].match(user_input)
        if match:
            groups=match.groups()
            slots={slot:groups[i].strip() for i, slot in enumerate(p["slots"])}
            result={"Intent":p["intent"]}
            result.update(slots)
            return result
    return {"Intent": "unknown", "raw_input":user_input}

def prepare_output_folder(folder="output"):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def save_to_file(data, filename="output/intent_1.json"): # Burada 1 yerine i yazılsa daha iteratif olur.
    with open(filename, "w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    print(f"JSON kaydedildi: {filename}")