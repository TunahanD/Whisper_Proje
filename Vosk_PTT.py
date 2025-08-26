import sounddevice as sd
import queue
import sys
import json
from vosk import Model, KaldiRecognizer

# === CONFIG ===
MODEL_PATH = "vosk-model-small-tr-0.22"

# === Model yükle ===
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)

# === Ses kuyruğu ===
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# === Mikrofon başlat ===
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                       channels=1, callback=callback):
    print("Konuşmaya başlayabilirsiniz (Çıkmak için CTRL+C)...")

    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            print(">>", res.get("text", ""))
        else:
            partial = json.loads(rec.PartialResult())
            # İstersen anlık tahmini de görebilirsin:
            # print("...", partial.get("partial", ""))
