"""
Skript pro sběr dat gest ruky pomocí webkamery.
Data jsou ukládána do CSV souboru pro pozdější trénování ML modelu.

Poznámka: Skript používá nové MediaPipe Tasks API (verze 0.10+),
          které nahradilo starší mp.solutions rozhraní.

Ovládání:
  's' ... začít/zastavit nahrávání snímků do CSV
  'q' ... ukončit program
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import os
import time


# ============================================================
#  NASTAVENÍ – měň pouze tuto sekci před každým nahráváním
# ============================================================

# Název gesta, které chceš nahrávat (label v CSV).
# Změň tuto hodnotu před každou novou sadou nahrávání.
NAZEV_GESTA = "posun dolu"          # příklady: "stop", "ok", "mír", "palec_nahoru"

# Cesta k výstupnímu CSV souboru.
# Pokud soubor neexistuje, skript ho vytvoří i s hlavičkou.
SOUBOR_CSV = "dataset.csv"

# Index kamery – většinou 0 pro vestavěnou webkameru.
INDEX_KAMERY = 0

# Minimální spolehlivost, při které MediaPipe vyhodnotí detekci jako platnou.
# Hodnota 0.0–1.0; vyšší = přísnější filtrování.
MIN_SPOLEHLIVOST_DETEKCE = 0.7
MIN_SPOLEHLIVOST_SLEDOVANI = 0.7

# ============================================================
#  KONEC NASTAVENÍ
# ============================================================


# ------------------------------------------------------------
# Stažení modelu detekce ruky (pokud ještě není na disku)
# ------------------------------------------------------------

# Cesta k souboru s natrénovaným modelem (*.task = formát MediaPipe Tasks).
MODEL_PATH = "hand_landmarker.task"

# URL ke stažení modelu z oficiálního úložiště Google / MediaPipe.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Zkontrolujeme, zda model existuje na disku.
# Pokud ne, program skončí s návodem, jak ho stáhnout.
if not os.path.isfile(MODEL_PATH):
    print("CHYBA: Soubor modelu '" + MODEL_PATH + "' nebyl nalezen.")
    print("Stáhni ho ručně příkazem:")
    print("  curl -L -o hand_landmarker.task \"" + MODEL_URL + "\"")
    exit()


# ------------------------------------------------------------
# Definice spojení kloubů ruky pro kreslení kostry
# ------------------------------------------------------------

# MediaPipe čísluje 21 bodů ruky (0 = zápěstí, 1–20 = klouby prstů).
# Každá dvojice čísel (a, b) znamená: nakresli čáru mezi bodem a a bodem b.
# Tím vznikne vizuální kostra ruky.
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # palec
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # ukazovák
    (0, 9),  (9, 10), (10, 11),(11, 12),   # prostředník
    (0, 13),(13, 14), (14, 15),(15, 16),   # prsteník
    (0, 17),(17, 18), (18, 19),(19, 20),   # malík
    (5, 9),  (9, 13), (13, 17)             # příčné spojení dlaně
]


# ------------------------------------------------------------
# Inicializace detektoru ruky (MediaPipe Tasks API)
# ------------------------------------------------------------

# BaseOptions říká detektoru, kde najde soubor s modelem.
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

# HandLandmarkerOptions nastavuje parametry detektoru.
#   running_mode=VIDEO → zpracováváme kontinuální video (ne jednotlivé snímky),
#                        detektor si pak udržuje stav a sleduje ruku průběžně.
#   num_hands=1        → hledáme maximálně jednu ruku.
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=MIN_SPOLEHLIVOST_DETEKCE,
    min_hand_presence_confidence=MIN_SPOLEHLIVOST_SLEDOVANI,
    min_tracking_confidence=MIN_SPOLEHLIVOST_SLEDOVANI
)

# Vytvoříme samotný detektor ze zadaných možností.
detektor = mp_vision.HandLandmarker.create_from_options(options)


# ------------------------------------------------------------
# Příprava CSV souboru
# ------------------------------------------------------------

# Sestavíme hlavičku CSV souboru.
# První sloupec bude "label" (název gesta).
# Pak pro každý z 21 bodů ruky přidáme tři sloupce: x0, y0, z0, x1, y1, z1, ...
hlavicka = ["label"]
for i in range(21):
    hlavicka.append("x" + str(i))
    hlavicka.append("y" + str(i))
    hlavicka.append("z" + str(i))
# Výsledná hlavička má 1 + 21*3 = 64 sloupců.

# Zjistíme, zda CSV soubor již existuje.
# Pokud ne, vytvoříme ho a zapíšeme hlavičku, aby byl soubor validní.
soubor_existuje = os.path.isfile(SOUBOR_CSV)

# Otevřeme soubor v režimu "a" (append = přidávat na konec),
# takže se data z předchozích nahrávání neztratí.
# newline="" je nutné na Windows, aby csv modul nevkládal prázdné řádky.
csv_soubor = open(SOUBOR_CSV, mode="a", newline="", encoding="utf-8")
csv_zapisovac = csv.writer(csv_soubor)

# Pokud soubor právě vznikl, zapíšeme hlavičku jako první řádek.
if not soubor_existuje:
    csv_zapisovac.writerow(hlavicka)


# ------------------------------------------------------------
# Otevření webkamery
# ------------------------------------------------------------

# cv2.VideoCapture otevře stream z kamery daného indexu.
kamera = cv2.VideoCapture(INDEX_KAMERY)

# Ověříme, že se kameru podařilo otevřít.
# Pokud ne, skript oznámí chybu, uklidí zdroje a skončí.
if not kamera.isOpened():
    print("CHYBA: Nelze otevřít kameru s indexem", INDEX_KAMERY)
    csv_soubor.close()
    detektor.close()
    exit()


# ------------------------------------------------------------
# Stavové proměnné programu
# ------------------------------------------------------------

# Příznak nahrávání: True = data se zapisují do CSV, False = čekáme.
nahravani_aktivni = False

# Počítadlo uložených snímků – zobrazíme ho v okně pro přehled.
pocet_ulozenych = 0

# Čas spuštění programu v sekundách – slouží k výpočtu časových razítek.
# MediaPipe VIDEO mód vyžaduje, aby každý snímek měl unikátní
# a stále rostoucí časové razítko v milisekundách.
cas_start = time.time()


# ------------------------------------------------------------
# Hlavní smyčka – zpracování videa snímek po snímku
# ------------------------------------------------------------

print("Program spuštěn.")
print("  Gesto k nahrání:", NAZEV_GESTA)
print("  Stiskni 's' pro zahájení/zastavení nahrávání.")
print("  Stiskni 'q' pro ukončení programu.")

while True:
    # Načteme jeden snímek z kamery.
    # 'uspech' je True/False, 'snimek' je numpy pole s pixely.
    uspech, snimek = kamera.read()

    # Pokud se čtení nezdaří (např. kamera se odpojila), skončíme smyčku.
    if not uspech:
        print("CHYBA: Nepodařilo se načíst snímek z kamery.")
        break

    # Horizontálně překlopíme obraz, aby fungoval jako „zrcadlo".
    # Bez toho by se pohyby levé/pravé ruky zobrazovaly obráceně.
    snimek = cv2.flip(snimek, 1)

    # MediaPipe pracuje s obrázky ve formátu RGB, ale OpenCV používá BGR.
    # Proto musíme barvy přehodit před zpracováním.
    snimek_rgb = cv2.cvtColor(snimek, cv2.COLOR_BGR2RGB)

    # Vypočteme časové razítko aktuálního snímku v milisekundách.
    # Hodnota musí být celé číslo a musí stále růst (nesmí se opakovat).
    timestamp_ms = int((time.time() - cas_start) * 1000)

    # Zabalíme numpy pole do objektu mp.Image, který Tasks API vyžaduje.
    # image_format=SRGB říká, že data jsou ve formátu RGB.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=snimek_rgb)

    # Spustíme detekci ruky na aktuálním snímku.
    # detect_for_video() je určeno pro VIDEO mód – využívá mezipaměti
    # pro plynulé sledování ruky mezi snímky.
    vysledek = detektor.detect_for_video(mp_image, timestamp_ms)


    # --------------------------------------------------------
    # Zpracování detekovaných bodů ruky
    # --------------------------------------------------------

    # Zkontrolujeme, zda MediaPipe v tomto snímku nějakou ruku nalezl.
    # hand_landmarks je seznam nalezených rukou; každá ruka je seznam 21 bodů.
    if vysledek.hand_landmarks:

        # Vezmeme první (a jedinou, protože num_hands=1) detekovanou ruku.
        ruka_body = vysledek.hand_landmarks[0]

        # Zjistíme rozměry snímku, abychom mohli převést normalizované
        # souřadnice (hodnoty 0.0–1.0) na pixelové souřadnice pro kreslení.
        vyska, sirka, _ = snimek.shape

        # Nakreslíme čáry kostry ruky (spojnice kloubů).
        for start_idx, end_idx in HAND_CONNECTIONS:
            # Získáme oba krajní body aktuální spojnice.
            start_bod = ruka_body[start_idx]
            end_bod   = ruka_body[end_idx]
            # Převedeme normalizované souřadnice (0.0–1.0) na pixely.
            start_px = (int(start_bod.x * sirka), int(start_bod.y * vyska))
            end_px   = (int(end_bod.x * sirka),   int(end_bod.y * vyska))
            # Nakreslíme bílou čáru mezi oběma body.
            cv2.line(snimek, start_px, end_px, (255, 255, 255), 2)

        # Nakreslíme zelené tečky na každý kloub ruky.
        for bod in ruka_body:
            px = (int(bod.x * sirka), int(bod.y * vyska))
            cv2.circle(snimek, px, 5, (0, 255, 0), -1)

        # Pokud je nahrávání aktivní, uložíme souřadnice do CSV.
        if nahravani_aktivni:

            # Začneme sestavovat řádek dat: první položka je název gesta.
            radek = [NAZEV_GESTA]

            # Projdeme všechny body ruky a přidáme jejich souřadnice.
            # Každý bod má x, y, z normalizované na rozsah 0.0–1.0.
            # Z je relativní hloubka (ne absolutní vzdálenost v cm).
            for bod in ruka_body:
                radek.append(bod.x)
                radek.append(bod.y)
                radek.append(bod.z)

            # Zapíšeme kompletní řádek (1 label + 63 čísel) do CSV souboru.
            csv_zapisovac.writerow(radek)

            # Okamžitě vyprázdníme vyrovnávací paměť, aby data nebyla ztracena
            # při neočekávaném ukončení programu.
            csv_soubor.flush()

            # Zvýšíme počítadlo uložených snímků.
            pocet_ulozenych = pocet_ulozenych + 1


    # --------------------------------------------------------
    # Vykreslení informačního textu do okna
    # --------------------------------------------------------

    # Sestavíme text, který zobrazí aktuální stav nahrávání.
    if nahravani_aktivni:
        stav_text = "NAHRAVA SE: " + NAZEV_GESTA + "  [snimky: " + str(pocet_ulozenych) + "]"
        barva_textu = (0, 0, 255)    # červená (BGR formát) = probíhá nahrávání
    else:
        stav_text = "POZASTAVENO – stiskni 'S' pro start"
        barva_textu = (0, 255, 0)    # zelená = čekáme na pokyn

    # Vypíšeme stavový text v horní části okna.
    cv2.putText(snimek, stav_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, barva_textu, 2)

    # Vypíšeme nápovědu pro klávesy v dolní části okna.
    cv2.putText(snimek, "'S' = start/stop  |  'Q' = konec", (10, snimek.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Zobrazíme snímek v okně s názvem "Sbírání dat".
    cv2.imshow("Sbírání dat", snimek)


    # --------------------------------------------------------
    # Obsluha kláves
    # --------------------------------------------------------

    # cv2.waitKey(1) čeká 1 ms na stisk klávesy a vrátí kód stisknuté klávesy.
    # Operace & 0xFF ořízne případné horní bity (nutné na některých systémech).
    klavesa = cv2.waitKey(1) & 0xFF

    # Klávesa 'q' (quit) → bezpečně ukončíme program.
    if klavesa == ord("q"):
        print("Ukončuji program...")
        break

    # Klávesa 's' (start/stop) → přepínáme stav nahrávání.
    if klavesa == ord("s"):
        nahravani_aktivni = not nahravani_aktivni   # přepnutí True ↔ False
        if nahravani_aktivni:
            print("Nahrávání ZAHÁJENO – gesto:", NAZEV_GESTA)
        else:
            print("Nahrávání POZASTAVENO. Celkem uloženo snímků:", pocet_ulozenych)


# ------------------------------------------------------------
# Úklid po ukončení smyčky
# ------------------------------------------------------------

# Uvolníme kameru, aby ji mohly použít jiné programy.
kamera.release()

# Zavřeme všechna OpenCV okna.
cv2.destroyAllWindows()

# Uzavřeme CSV soubor – tím zajistíme, že všechna data jsou skutečně zapsána na disk.
csv_soubor.close()

# Uvolníme detektor a jeho prostředky.
detektor.close()

print("Program ukončen. Data uložena do souboru:", SOUBOR_CSV)
print("Celkem uloženo snímků v této relaci:", pocet_ulozenych)
