import subprocess
import keyboard
import whisper
import time
import os
import platform
import re

filename = "kayit.wav"

# ----------------- Mikrofon Adını Otomatik Bulma -----------------
def get_default_microphone():
    system = platform.system()
    
    if system == "Windows":
        # ffmpeg ile cihaz listesini al
        result = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stderr=subprocess.PIPE, text=True
        )
        matches = re.findall(r'"([^"]+)"', result.stderr)
        # Mikrofonu filtrele
        audio_devices = [m for m in matches if "mikrofon" in m.lower() or "microphone" in m.lower()]
        if audio_devices:
            return audio_devices[0]  # İlk bulunan mikrofon
        return None

    elif system == "Linux":
        return "default"  # Alsa varsayılan

    elif system == "Darwin":  # macOS
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stderr=subprocess.PIPE, text=True
        )
        matches = re.findall(r'\[(\d+)\] (.+)', result.stderr)
        for idx, name in matches:
            if "microphone" in name.lower():
                return idx
        return "0"

    else:
        raise RuntimeError("Desteklenmeyen işletim sistemi.")

# ----------------- Ana Program -----------------
mic_name = get_default_microphone()
if mic_name is None:
    raise RuntimeError("Mikrofon bulunamadı!")

print(f"🎤 Kullanılacak mikrofon: {mic_name}")
print("Program hazır. Enter ile kaydı başlat, Esc ile durdur.")

ffmpeg_process = None

while True:
    # ENTER ile kayıt başlat
    if keyboard.is_pressed('enter') and ffmpeg_process is None:
        print("🎙 Kayıt başladı...")
        system = platform.system()
        if system == "Windows":
            cmd = ["ffmpeg", "-y", "-f", "dshow", "-i", f"audio={mic_name}", "-ar", "16000", "-ac", "1", filename]
        elif system == "Linux":
            cmd = ["ffmpeg", "-y", "-f", "alsa", "-i", mic_name, "-ar", "16000", "-ac", "1", filename]
        elif system == "Darwin":
            cmd = ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{mic_name}", "-ar", "16000", "-ac", "1", filename]
        else:
            raise RuntimeError("Desteklenmeyen işletim sistemi.")

        ffmpeg_process = subprocess.Popen(cmd)
        time.sleep(0.5)  # Tetiklemeyi engellemek için

    # ESC ile kayıt durdur
    if keyboard.is_pressed('esc') and ffmpeg_process is not None:
        print("⏹ Kayıt durdu. Dosya kapatılıyor...")
        ffmpeg_process.terminate()
        ffmpeg_process = None

        # Whisper ile transkript
        print("📜 Whisper transkripsiyon başlıyor...")
        model = whisper.load_model("medium")  # 'tiny', 'base', 'small', 'medium', 'large'
        result = model.transcribe(filename, language="tr")
        print("📝 Transkript:")
        print(result["text"])

        # Kaydı silmek istersen
        os.remove(filename)
        break
