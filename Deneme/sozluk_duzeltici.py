import re
from typing import List, Tuple, Dict

# ========== 1) SÖZLÜK ==========
# Kanonik -> varyantlar (yalın örnek; genişleteceğiz)
# Şablon Formatı -> Sol taraf istenen | Sağ taraf konuşmada söylenebilecek olan kelimeler.
terimSozluk: Dict[str, List[str]] = {
    # Kısaltmalar / cihazlar
    "PTZ": ["peteze", "pe te ze", "p t z", "pteze", "pte z"],
    "PTZ1": ["ptz 1", "ptz bir", "peteze bir"],
    "PTZ2": ["ptz 2", "ptz iki", "peteze iki", "ptz iki"],
    "PTZ3": ["ptz 3", "ptz üç", "peteze üç", "pe te ze üç"],
    "NVR": ["en ve ar", "enver", "n v r"],
    "IR":  ["ay ar", "ir", "infrared"],
    "tel-19": ["telon 9", "telon dokuz", "telan 9", "telan dokuz"],

    # Mod / özellik
    "geceModu": ["gece modu", "ir modu", "gece görüşü"],
    "dark mode":["dark modu","koyu mode","koyu modu"],
    "light mode":["light modu","aydınlık mode","aydınlık modu"],
    "hareketli": ["hareketliye", "hareketli moda", "devriye", "patrol", "motion"],
    "preset": ["prizet", "pirezset", "önayar"],
    "preset-1":["prizet bire","prizet 1'e","pirezset bire","pirezset 1'e","önayar bire","önayar 1'e","preset bire","preset 1'e"],

    # Düzen kalıpları (bazılarını kuralla da yakalıyoruz)
    "5x5": ["beşe beşlik", "beşe beş", "beş e beş", "5 e 5", "5’e 5", "5e5"],
    "3x3": ["üçe üçlük", "üçe üç", "üç e üç", "3 e 3", "3’e 3", "3e3"],

    # TR/EN karışan yerleşik terimler
    "blok": ["block", "blok"],
    "koridor": ["corridor", "koridor"],
    "arkaKoridor": ["arka koridor", "arka corridor", "arka korıdor"],

    # Kanal / CH varyantları (örnek)
    "kanal": ["ch", "channel", "kanal", "çeyç"],   # "çeyç" tipik telaffuz hatası
}

# Fiiller/yardımcılar: DOKUNMA (bunlar sadece korunur; normalize etmeyiz)
fiilKumesi = {
    "aç","kapat","oynat","duraklat","durdur","geri","ileri",
    "geç","çevir","al","ayarla","başlat","bitir","yakınlaştır","uzaklaştır"
}

# Harf adları -> harf (TR alfabe)
harfAdlari = {
    "a":"a","be":"b","ce":"c","çe":"ç","de":"d","e":"e","fe":"f","ge":"g","ğe":"ğ",
    "he":"h","ı":"ı","i":"i","je":"j","ke":"k","le":"l","me":"m","ne":"n","o":"o",
    "ö":"ö","pe":"p","re":"r","se":"s","şe":"ş","te":"t","u":"u","ü":"ü","ve":"v",
    "ye":"y","ze":"z"
}

# 0–25 sayı sözlüğü
sayiSozluk = {
    "sıfır":0, "bir":1,"iki":2,"üç":3,"dört":4,"beş":5,"altı":6,"yedi":7,"sekiz":8,"dokuz":9,
    "on":10,"on bir":11,"on iki":12,"on üç":13,"on dört":14,"on beş":15,
    "on altı":16,"on yedi":17,"on sekiz":18,"on dokuz":19,
    "yirmi":20,"yirmi bir":21,"yirmi iki":22,"yirmi üç":23,"yirmi dört":24,"yirmi beş":25
}

def trKucuk(s: str) -> str:
    return s.lower().replace("I","ı").replace("İ","i")

# ========== 2) ÖN-İŞLEME ==========
def normalizeBosluk(m: str) -> str:
    return re.sub(r"\s+", " ", m).strip()

def harfAdlariniHarfeCevir(m: str) -> str:
    # "pe te ze üç" -> "p t z üç"
    tokens = m.split()
    out = []
    for t in tokens:
        out.append(harfAdlari.get(t, t))
    return " ".join(out)

def turkceSayilariRakamaCevir(m: str) -> str:
    # önce iki kelimelikler
    for ikiK in [k for k in sayiSozluk if " " in k]:
        if ikiK in m:
            m = m.replace(ikiK, str(sayiSozluk[ikiK]))
    # tek kelimelikler
    for k, v in sayiSozluk.items():
        if " " not in k:
            m = re.sub(rf"\b{k}\b", str(v), m)
    return m

def gridKural(m: str) -> str:
    # 5e5 / 5 e 5 / 5’e 5 -> 5x5
    m = re.sub(r"\b(\d+)\s*[x×]\s*(\d+)\b", r"\1x\2", m)
    m = re.sub(r"\b(\d+)\s*(?:e|'?e)\s*(\d+)\b", r"\1x\2", m)
    return m

def dakikaBirimKural(m: str) -> str:
    # "10 dakikalık" / "10 dakika" -> "10 dk"
    m = re.sub(r"\b(\d+)\s*dakika(lık)?\b", r"\1 dk", m)
    return m

def blokYazimKural(m: str) -> str:
    # "a blok" / "A block" -> "A blok"
    # harf + blok: harfi büyük yaz
    m = re.sub(r"\b([a-zA-Z])\s*(block|blok)\b", lambda mo: f"{mo.group(1).upper()} blok", m)
    # çıplak "block" -> "blok"
    m = re.sub(r"\bblock\b", "blok", m)
    return m


def varyantHarita(lex: Dict[str, List[str]]) -> Dict[str, str]:
    h = {}
    for canon, vars_ in lex.items():
        h[trKucuk(canon)] = canon
        for v in vars_:
            h[trKucuk(v)] = canon
    return h

v2k = varyantHarita(terimSozluk)


# ========== 4) ANA DÜZELTİCİ ==========
def sozlukDuzelt2(metin: str) -> str:
    raw = metin
    m = trKucuk(metin)
    m = normalizeBosluk(m)
    m = harfAdlariniHarfeCevir(m)
    m = turkceSayilariRakamaCevir(m)
    m = gridKural(m)
    m = dakikaBirimKural(m)
    m = blokYazimKural(m)

    tokens = m.split()
    i = 0
    sonuc = []

    while i < len(tokens):
        eslendi = False

        # fiilleri aynen geçir; fuzzy'ye sokma
        if tokens[i] in fiilKumesi:
            sonuc.append(tokens[i])
            i += 1
            continue

        for n in [3,2,1]:
            if i+n <= len(tokens):
                parca = " ".join(tokens[i:i+n])
                p = trKucuk(parca)

                # 1) doğrudan eşleşme
                if p in v2k:
                    sonuc.append(v2k[p])
                    i += n
                    eslendi = True
                    break
                                
        if not eslendi:
            sonuc.append(tokens[i])
            i += 1

    out = " ".join(sonuc)

    # Stil: PTZ + sayı bitişik ve büyük harf
    out = re.sub(r"\bptz\s*(\d+)\b", lambda mo: f"PTZ{mo.group(1)}", out)
    out = out.replace("ptz", "PTZ")  # tek başına geçtiyse
    # kanal 13 -> kanal13, ch 13 -> kanal13
    out = re.sub(r"\b(ch|channel|kanal)\s*(\d+)\b", r"kanal\2", out)

    return normalizeBosluk(out)

# ========== 5) SÖZLÜĞE KOLAY EKLEME ==========
def sozlugeEkle(kanonik: str, *varyantlar: str):
    # runtime güncelleme
    if kanonik not in terimSozluk:
        terimSozluk[kanonik] = []
    for v in varyantlar:
        if v not in terimSozluk[kanonik]:
            terimSozluk[kanonik].append(v)
    global v2k
    v2k = varyantHarita(terimSozluk)



