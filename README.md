# Gesture Remote Controller

Aplikace pro ovládání počítače gesty ruky snímanými webkamerou.
Rozpoznává 5 gest pomocí modelu strojového učení (Random Forest) natrénovaného
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
(odešle stisk klávesy / scroll myši)
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
│   └── dataset.csv          # Nasbíraná trénovací data (4 999 snímků)
├── models/
│   ├── hand_landmarker.task # MediaPipe model detekce ruky (předtrénovaný, Google)
│   ├── model.pkl            # Natrénovaný Random Forest (generuje ml/train.py)
│   ├── scaler.pkl           # StandardScaler (generuje ml/train.py)
│   └── label_encoder.pkl    # LabelEncoder (generuje ml/train.py)
├── notebooks/
│   └── train_model.ipynb    # Google Colab notebook – celý ML postup
├── tests/                   # Testovací sada (unit testy)
├── run.py                   # Vstupní bod aplikace
├── start.bat                # Spouštěč pro Windows (instalace + spuštění jedním klikem)
├── pyproject.toml           # Konfigurace projektu a nástrojů
└── requirements.txt         # Závislosti
```

> **Oddělení kódu třetích stran:** Veškerý kód třetích stran (mediapipe, opencv,
> scikit-learn, pynput, pystray, …) je instalován jako balíčky do `venv/`
> a oddělen od autorského kódu v `gesture/`, `ml/` a `gui/`.
> Žádný cizí kód se nenachází přímo v souborech aplikace.

---

## Požadavky

- **Python 3.9–3.11** (mediapipe nepodporuje Python 3.12 a novější)
- Webkamera (vestavěná nebo USB)
- Windows / macOS / Linux

---

## Instalace a spuštění

### Windows – rychlé spuštění (bez IDE)

Dvakrát klikni na soubor **`start.bat`** v kořenovém adresáři projektu.

Skript automaticky:
1. Ověří instalaci Pythonu a vypíše verzi
2. Vytvoří virtuální prostředí `venv/`
3. Nainstaluje všechny závislosti
4. Spustí aplikaci

Při dalším spuštění se přeskočí kroky, které jsou již hotové.

> Pokud `pip install` selže, okno zůstane otevřené a zobrazí pokyny k opravě.
> Nejčastější příčina: Python 3.12 nebo novější – nainstaluj Python 3.11.

### Ruční instalace (všechny platformy)

Všechny příkazy spouštěj z **kořenového adresáře projektu** (`Gesture/`).

#### 1. Vytvoření virtuálního prostředí

```bash
python -m venv venv
```

#### 2. Aktivace virtuálního prostředí

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

#### 3. Instalace závislostí

```bash
pip install -r requirements.txt
```

#### 4. Trénování modelu

```bash
python ml/train.py
```

Výstup: `models/model.pkl`, `models/scaler.pkl`, `models/label_encoder.pkl`

#### 5. Spuštění aplikace

**Normální režim** (okno s obrazem kamery):
```bash
python run.py
```
Otevře se okno s obrazem z webkamery. Drž ruku před kamerou a prováděj gesta.
Ukončení: stiskni klávesu `Q` v okně aplikace.

**Režim na pozadí** (ikona v liště, nahrazuje trackpad):
```bash
python run.py --background
```
Žádné okno. Ikona se zobrazí v menu liště (macOS) nebo systémové liště (Windows/Linux).
Ukončení: pravý klik na ikonu → Ukončit.

#### 6. Automatické spouštění při přihlášení (volitelné)

```bash
python scripts/install.py
```

Aplikace se od této chvíle spustí automaticky na pozadí při každém přihlášení.
Odinstalace: `python scripts/uninstall.py`

---

## Průvodce nastavením (GUI)

Grafický průvodce nastavením lze spustit příkazem:

```bash
python gui/wizard.py
```

Umožňuje:
- Vizuálně upravit všechna nastavení pomocí posuvníků
- Spustit trénování modelu
- Zobrazit stav datasetu a modelových souborů
- Uložit nastavení do `settings.json`

---

## Sběr vlastních dat

Pokud chceš nasbírat nová data (například pro změnu nebo přidání gesta):

```bash
python ml/collect.py "nazev_gesta"
```

Příklad:
```bash
python ml/collect.py "posun doleva"
python ml/collect.py "posun doprava"
```

Ovládání sběru:
- `S` – zahájit / zastavit nahrávání
- `Q` – ukončit program

Data se přidají do `data/dataset.csv`. Po sběru je nutné model znovu natrénovat:

```bash
python ml/train.py
```

> Chceš-li gesto nahrát znovu od začátku, nejprve smaž jeho řádky z `data/dataset.csv`
> (všechny řádky, kde první sloupec odpovídá názvu gesta), pak smaž `models/model.pkl`
> a spusť sběr dat a trénování znovu.

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

Soubor `gesture/config.py` obsahuje nastavitelné konstanty. Hodnoty lze přepsat
bez úpravy zdrojového kódu pomocí souboru `settings.json` v kořenovém adresáři
(vytváří ho průvodce nastavením `gui/wizard.py`).

| Proměnná | Výchozí | Popis |
|---|---|---|
| `CAMERA_INDEX` | `0` | Index webkamery (0 = vestavěná) |
| `DETECTION_CONFIDENCE` | `0.7` | Minimální spolehlivost detekce ruky |
| `TRACKING_CONFIDENCE` | `0.7` | Minimální spolehlivost sledování ruky |
| `PREDICTION_THRESHOLD` | `0.75` | Minimální jistota modelu pro spuštění akce |
| `GESTURE_COOLDOWN` | `0.4` | Prodleva (s) mezi opakováním stejného gesta |
| `CONTROL_MODE` | `"scroll"` | Režim ovládání: `"scroll"` (myš) nebo `"keyboard"` (šipky) |
| `SCROLL_AMOUNT` | `10` | Počet jednotek scrollu na jedno gesto |

---

## O datech

Dataset obsahuje **4 999 snímků** pěti gest:

| Gesto | Počet snímků |
|---|---|
| posun nahoru | 1 061 |
| posun dolu | 1 142 |
| posun doprava | 1 111 |
| posun doleva | 1 106 |
| pauza | 579 |

**Formát:** CSV, 64 sloupců – `label` + `x0,y0,z0,...,x20,y20,z20`
(souřadnice 21 kloubů ruky normalizované na rozsah 0–1)

**Původ dat:** Data byla sesbírána vlastní rukou pomocí `ml/collect.py`.
MediaPipe HandLandmarker detekoval klouby ruky v každém snímku webkamery.
Nahrávání probíhalo v různých vzdálenostech od kamery a různých pozicích ruky.

---

## Předzpracování dat

Podrobný postup je zdokumentován v `notebooks/train_model.ipynb` a provádí ho i `ml/train.py`:

| Krok | Metoda | Popis |
|---|---|---|
| Čištění | `dropna()` | Odstranění řádků s NaN – nastane, když MediaPipe ruku nenašel |
| Kódování | `LabelEncoder` | Převod textových labelů na čísla (abecedně: doleva=0, doprava=1, dolu=2, nahoru=3, pauza=4) |
| Škálování | `StandardScaler` | Normalizace příznaků na průměr=0, odchylka=1; `fit` pouze na trénovacích datech, aby nedošlo k data leakage |

---

## Poznámka ke spuštění na Windows

MediaPipe vyžaduje **Python 3.11 nebo starší**. Python 3.12 a 3.13 nejsou podporovány.
Pokud `pip install` selže, zkontroluj verzi Pythonu příkazem `python --version`.

Pokud se zobrazí chyba týkající se OpenCV nebo Visual C++, nainstaluj
[Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
a zopakuj instalaci.
