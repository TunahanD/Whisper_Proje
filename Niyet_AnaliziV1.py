import re
from typing import Dict, List, Optional

# ======================
# 1) SÖZLÜKLER (Ontology)
# ======================
CAMERAS = [
    "tel19",
    "hareketli3",
    "a blok arka koridor",
    "c blok",
    "a blok esd koridor 2"
]

PRESETS = ["preset1", "preset2", "preset3"]

DUZENLER = ["5x5", "2x2", "3x3", "4x4"]

# ===========================
# 2) INTENT TANIMLARI
# ===========================
INTENTS = {
    "kamera_ac": {
        "keywords": ["kamera", "aç"],
        "params": ["kamera_id"]
    },
    "kamera_preset": {
        "keywords": ["ptz", "preset"],
        "params": ["kamera_id", "preset"]
    },
    "gecmis_ac": {
        "keywords": ["geçmiş", "aç"],
        "params": ["kamera_id", "sure"]
    },
    "ekran_duzeni": {
        "keywords": ["ekran düzeni", "düzenine dön"],
        "params": ["duzen"]
    },
    "tam_ekran": {
        "keywords": ["tam ekran"],
        "params": ["kamera_id?"]  # opsiyonel parametre
    },
    "screenshot": {
        "keywords": ["ekran görüntüsü al"],
        "params": ["kamera_id"]
    }
}

# ===========================
# 3) INTENT PARSER SINIFI
# ===========================
class IntentParser:
    def __init__(self, intents: Dict, cameras: List[str], presets: List[str], duzenler: List[str]):
        self.intents = intents
        self.cameras = cameras
        self.presets = presets
        self.duzenler = duzenler

    def parse(self, text: str) -> Dict:
        text = text.lower()
        result = {"intent": "bilinmiyor", "params": {}}

        for intent, data in self.intents.items():
            if all(keyword in text for keyword in data["keywords"]):
                result["intent"] = intent

                # Parametre yakalama
                if "kamera_id" in data["params"] or "kamera_id?" in data["params"]:
                    kamera = next((k for k in self.cameras if k in text), None)
                    if kamera:
                        result["params"]["kamera_id"] = kamera

                if "preset" in data["params"]:
                    preset = next((p for p in self.presets if p in text), None)
                    if preset:
                        result["params"]["preset"] = preset

                if "duzen" in data["params"]:
                    duzen = next((d for d in self.duzenler if d in text), None)
                    if duzen:
                        result["params"]["duzen"] = duzen

                if "sure" in data["params"]:
                    match = re.search(r"(\d+)\s*dakika", text)
                    if match:
                        result["params"]["sure"] = f"{match.group(1)} dakika"

                break  # intent bulundu, çık
        return result


# ===========================
# 4) TEST
# ===========================
parser = IntentParser(INTENTS, CAMERAS, PRESETS, DUZENLER)

test_sentences = [
    "tel19 kamera görüntüsünü aç",
    "hareketli3 ptz kamerasını preset1'e çevir",
    "a blok arka koridor kamerasının 10 dakikalık geçmiş görüntüsünü aç",
    "5x5 ekran düzenine dön",
    "c blok kamera görüntüsünü tam ekran yap",
    "geçerli düzeni tam ekran yap",
    "a blok esd koridor 2 ekran görüntüsü al",
    "3x3 ekran düzenine dön",
    "a blok arka koridor kamera görüntüsünü aç",
    "tel19 ekran görüntüsü al"
]

for s in test_sentences:
    print(f"\nCümle: {s}")
    print("Çıktı:", parser.parse(s))
