#intent_parser.py
import re
import json
import os
import shutil

patterns=[ # REGEX KURALLARI
    {
        "intent":"camera_open",
        "pattern":re.compile(r"^(.*?) kamera(?:yı|sı|sını|nın)?( (görüntü|görüntüsü|görüntüsünü|ekranını|yayınını))? (aç|getir|başlat)\.?$"),
        "slots":["camera_name"]
    },
    {
        "intent":"camera_rotate",
        "pattern":re.compile(r"^(.*?) kamera(?:sı|sını|yı|yi|sının|nın)? (.+?) (çevir|döndür)\.?$",re.IGNORECASE),
        "slots":["camera_name","value"]
    },
    {
        "intent":"review_open",
        "pattern":re.compile(r"^(.*?) kamera(?:sı|sının)? (.+?) (geçmişini|geçmiş görüntüsünü|geçmiş görüntüyü) (aç|getir)\.?$",re.IGNORECASE),
        "slots":["camera_name","value"]

    },
    {
        "intent":"screen_layout",
        "pattern":re.compile(r"^(.*?) (ekran düzenine|ekran düzenini) (dön|geç|aç)\.?$",re.IGNORECASE),   # FRONT END TARAFI
        "slots":["template_layout"]

    },
    {
        "intent":"full_screen",
        "pattern":re.compile(r"^(.*?) kamera(?:sı|sını)? (görüntüsünü|ekranını)? (tam ekran yap|tam ekrana al|tam ekrana getir|ekranı kapla)\.?$",re.IGNORECASE),   # FRONT END TARAFI
        "slots":["camera_name"]
    },
    {
        "intent":"screen_layout_full_screen",
        "pattern":re.compile(r"^(.*?) (ekran|düzen)(i|ini)? (tam ekran yap|tam ekrana al|büyüt)\.?$",re.IGNORECASE),         # FRONT END TARAFI
        "slots":["last_template_layout"]
    },
    {
        "intent":"screenshot",
        "pattern":re.compile(r"^(.*?) ekran görüntüsü(nü|n)? (al|çek|yakala|kaydet)\.?$",re.IGNORECASE),
        "slots":["camera_name"]
    },
    {
        "intent":"dark_mode",
        "pattern":re.compile(r"^(dark mode aç|karanlık modu aç|gece modunu aç|koyu mod aç|ekranı karanlık yap|gece temasına geç|light mode kapat)\.?$",re.IGNORECASE),          # FRONT END TARAFI => Light mode kapat dediğinde burasının açılması gerek!
        "slots":[]
    },
    {
        "intent":"light_mode",
        "pattern":re.compile(r"^(light mode aç|aydınlık modu aç|gündüz modunu aç|açık modu aç|ekranı aydınlık yap|gündüz temasına geç|dark mode kapat)\.?$",re.IGNORECASE),        # FRONT END TARAFI => Dark mode kapat dediğinde burasının açılması gerek!
        "slots":[]
    },
]

def parse_intent(user_input:str): # Kullanıcıdan gelen cümleyi niyet analizi yapar. Esnek regex desenlerine göre intent belirlenir ve ilgili slotlar doldurulur.
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

def prepare_output_folder(folder="output"): #  Çıktıların kaydedileceği klasörü hazırlar.  Eğer klasör varsa silinir ve yeniden oluşturulur.
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def save_to_file(data, filename="output/intent_1.json"): # Burada 1 yerine i yazılsa daha iteratif olur.
    with open(filename, "w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    print(f"JSON kaydedildi: {filename}")