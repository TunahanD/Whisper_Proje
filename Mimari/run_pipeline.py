from main import transcribe_wav, CFG
from intent_parser import parse_intent, prepare_output_folder, save_to_file

def main():
    print("Ses dosyası işleniyor: Transkripsiyon + Niyet Analizi")

    prepare_output_folder()

    audio_path="Kayıt.wav"

    CFG["audioFile"]=audio_path
    print(f"Transkripsiyon başlıyor: {audio_path}")
    raw_text=transcribe_wav(audio_path, return_text=True)
    print(f"Transkript edilen metin: {raw_text}")

    intent_result=parse_intent(raw_text)
    save_to_file(intent_result)

def process_audio_from_memory(audio_bytes_io):
    """Flask için BytesIO objesi alan versiyon"""
    print("Ses dosyası işleniyor: Transkripsiyon + Niyet Analizi")
    
    print("Transkripsiyon başlıyor...")
    raw_text = transcribe_wav(audio_bytes_io, return_text=True)
    print(f"Transkript edilen metin: {raw_text}")
    
    intent_result = parse_intent(raw_text)
    return intent_result  # ← Dosyaya kaydetmek yerine direkt return

if __name__ == "__main__":
    main()


