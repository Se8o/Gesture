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

| Gesto | Klávesa | Použití |
|---|---|---|
| posun nahoru | ↑ | Hlasitost nahoru, předchozí snímek |
| posun dolu | ↓ | Hlasitost dolů, další snímek |
| posun doprava | → | Vpřed, další strana prezentace |
| posun doleva | ← | Zpět, předchozí strana |

---

## Struktura projektu

```
Gesture/
├── src/                     # Zdrojový kód aplikace (autorský)
│   ├── config.py            # Konfigurační konstanty a cesty
│   ├── recognizer.py        # Detekce ruky + predikce gesta
│   ├── controller.py        # Převod gesta na stisk klávesy
│   └── app.py               # Hlavní smyčka aplikace
├── scripts/
│   └── collect_data.py      # Sběr trénovacích dat
├── data/
│   └── dataset.csv          # Nasbíraná trénovací data (4 117 snímků)
├── models/
│   ├── hand_landmarker.task # MediaPipe model detekce ruky (předtrénovaný Google)
│   ├── model.pkl            # Natrénovaný Random Forest (generuje train.py)
│   ├── scaler.pkl           # StandardScaler (generuje train.py)
│   └── label_encoder.pkl    # LabelEncoder (generuje train.py)
├── notebook/
│   └── train_model.ipynb    # Google Colab notebook – celý ML postup
├── run.py                   # Vstupní bod aplikace
├── train.py                 # Lokální trénování modelu
└── requirements.txt         # Závislosti
```

> **Poznámka k třetím stranám:** Veškerý kód třetích stran (mediapipe, opencv,
> scikit-learn, pynput, …) je instalován jako balíčky do `venv/`
> a oddělen od autorského kódu v `src/`. Žádný cizí kód se nenachází
> přímo v souborech aplikace.

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
python train.py
```

Výstup: `models/model.pkl`, `models/scaler.pkl`, `models/label_encoder.pkl`

### 5. Spuštění aplikace

```bash
python run.py
```

Otevře se okno s obrazem z webkamery. Drž ruku před kamerou a prováděj gesta.

**Ukončení:** stiskni klávesu `Q` v okně aplikace.

---

## Sběr vlastních dat

Pokud chceš nasbírat nová data (například pro rozšíření o další gesta):

```bash
# Nahraď "nazev_gesta" názvem gesta, které chceš nahrávat
python scripts/collect_data.py "nazev_gesta"
```

Ovládání sběru:
- `S` – zahájit / zastavit nahrávání
- `Q` – ukončit program

Data se přidají do `data/dataset.csv`. Po sběru je nutné model znovu natrénovat:

```bash
python train.py
```

---

## Trénování v Google Colab

Notebook `notebook/train_model.ipynb` lze spustit v Google Colab:

1. Otevři [Google Colab](https://colab.research.google.com/)
2. Nahraj soubor `notebook/train_model.ipynb`
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

---

## O datech

Dataset obsahuje **4 117 snímků** čtyř gest:

| Gesto | Počet snímků |
|---|---|
| posun nahoru | 1 045 |
| posun dolu | 1 026 |
| posun doprava | 1 023 |
| posun doleva | 1 023 |

**Formát:** CSV, 64 sloupců – `label` + `x0,y0,z0,...,x20,y20,z20`
(souřadnice 21 kloubů ruky normalizované na rozsah 0–1)

**Původ dat:** Data byla sesbírána vlastní rukou pomocí `scripts/collect_data.py`.
MediaPipe HandLandmarker detekoval klouby ruky v každém snímku webkamery.
Nahrávání probíhalo v různých vzdálenostech od kamery a různých pozicích ruky.
