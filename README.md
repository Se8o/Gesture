# Gesture Remote Controller

Aplikace pro ovládání počítače gesty ruky snímanými webkamerou.
Rozpoznává 4 gesta pomocí modelu strojového učení (Random Forest) natrénovaného
na datech nasbíraných vlastní rukou.

---

## Jak to funguje

```
Webkamera
   ↓
MediaPipe HandLandmarker
(detekuje 21 kloubů ruky)
   ↓
Random Forest Classifier
(předpoví gesto)
   ↓
pynput
(odešle stisk klávesy)
```

| Gesto | Akce | Použití |
|---|---|---|
| posun nahoru | ↑ / scroll nahoru | Hlasitost nahoru, předchozí snímek |
| posun dolu | ↓ / scroll dolů | Hlasitost dolů, další snímek |
| posun doprava | Ctrl+Tab | Přepnutí na pravý tab prohlížeče |
| posun doleva | Ctrl+Shift+Tab | Přepnutí na levý tab prohlížeče |
| pauza | Mezerník | Spuštění / pozastavení videa |

---

## Struktura projektu

```
Gesture/
├── gesture/                 # Hlavní balíček aplikace (autorský kód)
│   ├── app.py               # Hlavní smyčka aplikace
│   ├── config.py            # Konfigurační konstanty a cesty
│   ├── controller.py        # Převod gesta na akci (klávesa / scroll)
│   ├── recognizer.py        # Detekce ruky + predikce gesta
│   └── tray.py              # Ikona v systémové liště
├── ml/                      # ML pipeline (autorský kód)
│   ├── train.py             # Trénování modelu
│   └── collect.py           # Sběr trénovacích dat webkamerou
├── gui/                     # GUI nástroje (autorský kód)
│   └── wizard.py            # Průvodce nastavením (Tkinter)
├── scripts/                 # Údržbové skripty
│   ├── install.py           # Registrace autostartu
│   └── uninstall.py         # Odstranění autostartu
├── data/
│   └── dataset.csv          # Nasbíraná trénovací data (4 117 snímků)
├── models/
│   ├── hand_landmarker.task # MediaPipe model detekce ruky (předtrénovaný, Google)
│   ├── model.pkl            # Natrénovaný Random Forest (generuje ml/train.py)
│   ├── scaler.pkl           # StandardScaler (generuje ml/train.py)
│   └── label_encoder.pkl    # LabelEncoder (generuje ml/train.py)
├── notebooks/
│   └── train_model.ipynb    # Google Colab notebook – celý ML postup
├── tests/                   # Testovací sada (76 unit testů)
├── run.py                   # Vstupní bod aplikace
├── pyproject.toml           # Konfigurace projektu a nástrojů
└── requirements.txt         # Závislosti
```

> **Oddělení kódu třetích stran:** Veškerý kód třetích stran (mediapipe, opencv,
> scikit-learn, pynput, pystray, …) je instalován jako balíčky do `venv/`
> a oddělen od autorského kódu v `gesture/`, `ml/` a `gui/`.
> Žádný cizí kód se nenachází přímo v souborech aplikace.

---

## Požadavky

- Python 3.9 nebo novější
- Webkamera (vestavěná nebo USB)
- Windows / macOS / Linux

---

## Instalace a spuštění (bez IDE)

Všechny příkazy spouštěj z **kořenového adresáře projektu** (`Gesture/`).

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv venv
```

### 2. Aktivace virtuálního prostředí

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 4. Trénování modelu

```bash
python ml/train.py
```

Výstup: `models/model.pkl`, `models/scaler.pkl`, `models/label_encoder.pkl`

### 5. Spuštění aplikace

**Normální režim** (okno s obrazem kamery):
```bash
python run.py
```
Otevře se okno s obrazem z webkamery. Drž ruku před kamerou a prováděj gesta.
Ukončení: stiskni klávesu `Q` v okně aplikace.

**Režim na pozadí** (jako systémová extenze, nahrazuje trackpad):
```bash
python run.py --background
```
Žádné okno. Ikona se zobrazí v menu liště (macOS) nebo systémové liště (Windows/Linux).
Ukončení: pravý klik na ikonu → Ukončit.

### 6. Automatické spouštění při přihlášení (volitelné)

```bash
python scripts/install.py
```

Aplikace se od této chvíle spustí automaticky na pozadí při každém přihlášení.
Odinstalace: `python scripts/uninstall.py`

---

## Sběr vlastních dat

Pokud chceš nasbírat nová data (například pro rozšíření o další gesta):

```bash
# Nahraď "nazev_gesta" názvem gesta, které chceš nahrávat
# Příklad: python ml/collect.py "pauza"
python ml/collect.py "nazev_gesta"
```

Ovládání sběru:
- `S` – zahájit / zastavit nahrávání
- `Q` – ukončit program

Data se přidají do `data/dataset.csv`. Po sběru je nutné model znovu natrénovat:

```bash
python ml/train.py
```

---

## Trénování v Google Colab

Notebook `notebooks/train_model.ipynb` lze spustit v Google Colab:

1. Otevři [Google Colab](https://colab.research.google.com/)
2. Nahraj soubor `notebooks/train_model.ipynb`
3. Spusť první buňku – nainstaluje knihovny a požádá o nahrání `data/dataset.csv`
4. Spusť všechny buňky v pořadí
5. Po dokončení stáhni soubory `model.pkl`, `scaler.pkl`, `label_encoder.pkl`
6. Přesuň stažené soubory do složky `models/`

---

## Konfigurace

Soubor `src/config.py` obsahuje nastavitelné konstanty:

| Proměnná | Výchozí | Popis |
|---|---|---|
| `CAMERA_INDEX` | `0` | Index webkamery (0 = vestavěná) |
| `DETECTION_CONFIDENCE` | `0.7` | Minimální spolehlivost detekce ruky |
| `PREDICTION_THRESHOLD` | `0.75` | Minimální jistota modelu pro spuštění akce |
| `GESTURE_COOLDOWN` | `1.0` | Prodleva (s) mezi opakováním stejného gesta |
| `CONTROL_MODE` | `"scroll"` | Režim ovládání: `"scroll"` (myš) nebo `"keyboard"` (šipky) |
| `SCROLL_AMOUNT` | `5` | Počet jednotek scrollu na jedno gesto |

---

## O datech

Dataset obsahuje **4 696 snímků** pěti gest:

| Gesto | Počet snímků |
|---|---|
| posun nahoru | 1 045 |
| posun dolu | 1 026 |
| posun doprava | 1 023 |
| posun doleva | 1 023 |
| pauza | 579 |

**Formát:** CSV, 64 sloupců – `label` + `x0,y0,z0,...,x20,y20,z20`
(souřadnice 21 kloubů ruky normalizované na rozsah 0–1)

**Původ dat:** Data byla sesbírána vlastní rukou pomocí `scripts/collect_data.py`.
MediaPipe HandLandmarker detekoval klouby ruky v každém snímku webkamery.
Nahrávání probíhalo v různých vzdálenostech od kamery a různých pozicích ruky.

---

## Předzpracování dat

Podrobný postup je zdokumentován v `notebooks/train_model.ipynb` (Kroky 3, 4, 6) a provádí ho i `train.py`:

| Krok | Metoda | Popis |
|---|---|---|
| Čištění | `dropna()` | Odstranění řádků s NaN – nastane, když MediaPipe ruku nenašel |
| Kódování | `LabelEncoder` | Převod textových labelů na čísla (abecedně: doleva=0, doprava=1, dolu=2, nahoru=3) |
| Škálování | `StandardScaler` | Normalizace příznaků na průměr=0, odchylka=1; `fit` pouze na trénovacích datech, aby nedošlo k data leakage |

---

## Poznámka ke spuštění na Windows

Pokud `pip install` selže s chybou týkající se OpenCV nebo MediaPipe, nainstaluj
[Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
a zopakuj instalaci.
