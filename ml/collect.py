"""
Skript pro sběr dat gest ruky pomocí webkamery.
Data jsou ukládána do CSV souboru pro pozdější trénování ML modelu.

Poznámka: Skript používá nové MediaPipe Tasks API (verze 0.10+),
          které nahradilo starší mp.solutions rozhraní.

Spuštění (z kořenového adresáře projektu):
    python ml/collect.py "posun nahoru"
    python ml/collect.py "posun dolu"
    python ml/collect.py "posun doprava"
    python ml/collect.py "posun doleva"

Ovládání:
  's' ... začít/zastavit nahrávání snímků do CSV
  'q' ... ukončit program
"""

import sys
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import time

# ============================================================
#  Zpracování argumentu příkazové řádky
# ============================================================

# Název gesta lze zadat jako argument: python ml/collect.py "posun nahoru"
# Pokud argument není zadán, použije se výchozí hodnota.
if len(sys.argv) >= 2:
    NAZEV_GESTA = sys.argv[1]
else:
    NAZEV_GESTA = "posun dolu"   # výchozí hodnota – změň podle potřeby

# ============================================================
#  Cesty k souborům (absolutní, odvozené od tohoto souboru)
# ============================================================

# Kořenový adresář projektu je o dvě úrovně výše než tento skript
# (Gesture/ml/collect.py → Gesture/)
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cesta k předtrénovanému modelu detekce ruky
MODEL_PATH = os.path.join(ROOT_DIR, "models", "hand_landmarker.task")

# Cesta k výstupnímu CSV souboru se sesbíranými daty
SOUBOR_CSV = os.path.join(ROOT_DIR, "data", "dataset.csv")

# ============================================================
#  Konfigurační konstanty
# ============================================================

INDEX_KAMERY               = 0    # 0 = vestavěná webkamera
MIN_SPOLEHLIVOST_DETEKCE   = 0.7
MIN_SPOLEHLIVOST_SLEDOVANI = 0.7

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ============================================================
#  Kontrola existence souborů
# ============================================================

if not os.path.isfile(MODEL_PATH):
    print("CHYBA: Soubor modelu '" + MODEL_PATH + "' nebyl nalezen.")
    print("Stáhni ho ručně příkazem:")
    print("  curl -L -o models/hand_landmarker.task \"" + MODEL_URL + "\"")
    sys.exit(1)

# Vytvoříme složku data/, pokud ještě neexistuje
os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)

# ============================================================
#  Definice spojení kloubů ruky pro kreslení kostry
# ============================================================

# MediaPipe čísluje 21 bodů ruky (0 = zápěstí, 1–20 = klouby prstů).
# Každá dvojice čísel (a, b) znamená: nakresli čáru mezi bodem a a bodem b.
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # palec
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # ukazovák
    (0, 9),  (9, 10), (10, 11),(11, 12),   # prostředník
    (0, 13),(13, 14), (14, 15),(15, 16),   # prsteník
    (0, 17),(17, 18), (18, 19),(19, 20),   # malík
    (5, 9),  (9, 13), (13, 17)             # příčné spojení dlaně
]

# ============================================================
#  Inicializace detektoru ruky (MediaPipe Tasks API)
# ============================================================

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

# running_mode=VIDEO → detektor si udržuje stav a sleduje ruku průběžně
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=MIN_SPOLEHLIVOST_DETEKCE,
    min_hand_presence_confidence=MIN_SPOLEHLIVOST_SLEDOVANI,
    min_tracking_confidence=MIN_SPOLEHLIVOST_SLEDOVANI
)
detektor = mp_vision.HandLandmarker.create_from_options(options)

# ============================================================
#  Příprava CSV souboru
# ============================================================

# Hlavička CSV: label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20 (64 sloupců)
hlavicka = ["label"]
for i in range(21):
    hlavicka.append("x" + str(i))
    hlavicka.append("y" + str(i))
    hlavicka.append("z" + str(i))

soubor_existuje = os.path.isfile(SOUBOR_CSV)

# Otevřeme soubor v režimu "append" – data z předchozích relací se zachovají
csv_soubor    = open(SOUBOR_CSV, mode="a", newline="", encoding="utf-8")
csv_zapisovac = csv.writer(csv_soubor)

if not soubor_existuje:
    csv_zapisovac.writerow(hlavicka)

# ============================================================
#  Otevření webkamery
# ============================================================

kamera = cv2.VideoCapture(INDEX_KAMERY)
if not kamera.isOpened():
    print("CHYBA: Nelze otevřít kameru s indexem", INDEX_KAMERY)
    csv_soubor.close()
    detektor.close()
    sys.exit(1)

# ============================================================
#  Stavové proměnné
# ============================================================

nahravani_aktivni = False
pocet_ulozenych   = 0
cas_start         = time.time()

# ============================================================
#  Hlavní smyčka
# ============================================================

print("Program spuštěn.")
print("  Gesto k nahrání:", NAZEV_GESTA)
print("  Stiskni 's' pro zahájení/zastavení nahrávání.")
print("  Stiskni 'q' pro ukončení programu.")

while True:
    uspech, snimek = kamera.read()
    if not uspech:
        print("CHYBA: Nepodařilo se načíst snímek z kamery.")
        break

    # Horizontálně překlopíme obraz – funguje jako zrcadlo
    snimek     = cv2.flip(snimek, 1)
    snimek_rgb = cv2.cvtColor(snimek, cv2.COLOR_BGR2RGB)

    # Časové razítko musí být unikátní a stále rostoucí (požadavek VIDEO módu)
    timestamp_ms = int((time.time() - cas_start) * 1000)
    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=snimek_rgb)
    vysledek     = detektor.detect_for_video(mp_image, timestamp_ms)

    if vysledek.hand_landmarks:
        ruka_body       = vysledek.hand_landmarks[0]
        vyska, sirka, _ = snimek.shape

        # Kreslení bílé kostry ruky
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_bod = ruka_body[start_idx]
            end_bod   = ruka_body[end_idx]
            start_px  = (int(start_bod.x * sirka), int(start_bod.y * vyska))
            end_px    = (int(end_bod.x * sirka),   int(end_bod.y * vyska))
            cv2.line(snimek, start_px, end_px, (255, 255, 255), 2)

        # Zelené tečky na každý kloub
        for bod in ruka_body:
            px = (int(bod.x * sirka), int(bod.y * vyska))
            cv2.circle(snimek, px, 5, (0, 255, 0), -1)

        # Zápis souřadnic do CSV (jen při aktivním nahrávání)
        if nahravani_aktivni:
            # Řádek: název_gesta, x0, y0, z0, x1, y1, z1, ... (64 hodnot)
            radek = [NAZEV_GESTA]
            for bod in ruka_body:
                radek.append(bod.x)
                radek.append(bod.y)
                radek.append(bod.z)
            csv_zapisovac.writerow(radek)
            # Okamžité vyprázdnění bufferu – data se neztrátí při pádu programu
            csv_soubor.flush()
            pocet_ulozenych = pocet_ulozenych + 1

    # Informační text v okně
    if nahravani_aktivni:
        stav_text   = "NAHRAVA SE: " + NAZEV_GESTA + "  [snimky: " + str(pocet_ulozenych) + "]"
        barva_textu = (0, 0, 255)    # červená = probíhá nahrávání
    else:
        stav_text   = "POZASTAVENO – stiskni 'S' pro start"
        barva_textu = (0, 255, 0)    # zelená = čekáme na pokyn

    cv2.putText(snimek, stav_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, barva_textu, 2)
    cv2.putText(snimek, "'S' = start/stop  |  'Q' = konec", (10, snimek.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Sbirani dat – " + NAZEV_GESTA, snimek)

    klavesa = cv2.waitKey(1) & 0xFF
    if klavesa == ord("q"):
        print("Ukončuji program...")
        break
    if klavesa == ord("s"):
        nahravani_aktivni = not nahravani_aktivni
        if nahravani_aktivni:
            print("Nahrávání ZAHÁJENO – gesto:", NAZEV_GESTA)
        else:
            print("Nahrávání POZASTAVENO. Celkem uloženo snímků:", pocet_ulozenych)

# ============================================================
#  Úklid po ukončení smyčky
# ============================================================

kamera.release()
cv2.destroyAllWindows()
csv_soubor.close()
detektor.close()

print("Program ukončen. Data uložena do souboru:", SOUBOR_CSV)
print("Celkem uloženo snímků v této relaci:", pocet_ulozenych)
