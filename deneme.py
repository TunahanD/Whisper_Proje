import whisper
import os

def transcribe_audio(audio_file_path, model_size="base", language="tr"):
    """
    Ses dosyasını metne çeviren fonksiyon
    
    Args:
        audio_file_path (str): Ses dosyasının yolu
        model_size (str): Whisper model boyutu ("tiny", "base", "small", "medium", "large")
        language (str): Dil kodu ("tr" Türkçe için)
    
    Returns:
        dict: Transkripsiyon sonuçları
    """
    
    print(f"Model yükleniyor: {model_size}")
    # Modeli yükle (ilk seferinde internet üzerinden indirilir)
    model = whisper.load_model(model_size)
    
    print(f"Ses dosyası işleniyor: {audio_file_path}")
    
    # Ses dosyasını transkript et
    # language="tr" parametresi Türkçe için optimize eder
    result = model.transcribe(audio_file_path, language=language)
    
    return result

def main():
    # Örnek kullanım
    audio_file = r"C:\Users\tdone\Desktop\ses.mp3"  # Kendi ses dosyanızın yolunu girin
    
    # Dosya var mı kontrol et
    if not os.path.exists(audio_file):
        print(f"Hata: {audio_file} dosyası bulunamadı!")
        print("Lütfen geçerli bir ses dosyası yolu belirtin.")
        return
    
    try:
        # En küçük model ile başlayalım (tiny)
        result = transcribe_audio(audio_file, model_size="tiny", language="tr")
        
        # Sonuçları göster
        print("\n" + "="*50)
        print("TRANSKRİPSİYON SONUÇLARI")
        print("="*50)
        print(f"Metin: {result['text']}")
        print(f"Algılanan Dil: {result['language']}")
        
        # Zaman damgaları ile detaylı sonuçlar
        print("\nDetaylı Sonuçlar (Zaman Damgalı):")
        print("-"*30)
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            print(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}")
            
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    main()