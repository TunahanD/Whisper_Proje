import os
import sys
import json
import wave
import subprocess
from vosk import Model, KaldiRecognizer

# === CONFIG ===
MODEL_PATH = r"C:\Users\tdone\Desktop\vosk-model-small-tr-0.3"  # Türkçe model klasörü
AUDIO_PATH = "Kayıt.wav"                 # Girdi ses dosyası (her format olabilir)
TMP_WAV = "converted.wav"                # ffmpeg ile dönüştürülmüş dosya

# === 1. ffmpeg ile uygun formata dönüştür ===
# -ar 16000 : 16kHz
# -ac 1     : Mono
# -f wav    : Çıktı formatı WAV
subprocess.run([
    "ffmpeg", "-y", "-i", AUDIO_PATH,
    "-ar", "16000", "-ac", "1", "-f", "wav", TMP_WAV
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === 2. Modeli yükle ===
if not os.path.exists(MODEL_PATH):
    print("Model bulunamadı, lütfen indirip MODEL_PATH değişkenine yolunu giriniz.")
    sys.exit(1)

model = Model(MODEL_PATH)

# === 3. Ses dosyasını aç ===
wf = wave.open(TMP_WAV, "rb")
rec = KaldiRecognizer(model, wf.getframerate())

# === 4. Transkript oluştur ===
result_text = ""

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        res = json.loads(rec.Result())
        result_text += " " + res.get("text", "")

# Son kalan parçayı ekle
final_res = json.loads(rec.FinalResult())
result_text += " " + final_res.get("text", "")

print("\n=== Transkript ===")
print(result_text.strip())
