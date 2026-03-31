# Mini-Problems Pipeline

Pipeline do generowania, rozwiązywania i ewaluacji mini-problemów kreatywności.
Zbudowany na wzorcu `run_reg.py` — używa **vLLM** do inferencji i **Hydra** do konfiguracji.

## Struktura projektu

```
mini-problems/
├── configs/
│   ├── config-generate.yaml     # Konfiguracja generatora
│   ├── config-solve.yaml        # Konfiguracja solvera
│   ├── config-evaluate.yaml     # Konfiguracja ewaluatora
│   └── model/
│       ├── llama3-8b.yaml       # Llama 3 8B (domyślny)
│       ├── llama3-70b.yaml      # Llama 3 70B
│       └── deepseek.yaml        # DeepSeek V2
├── prompts/
│   ├── generate.json            # Prompty do generowania problemów
│   ├── solve.json               # Prompty do rozwiązywania
│   └── evaluate.json            # Prompty do ewaluacji
├── data/
│   └── examples.json            # 11 przykładowych mini-problemów
├── run_generate.py              # Moduł 1: Generator
├── run_solve.py                 # Moduł 2: Solver
├── run_evaluate.py              # Moduł 3: Evaluator
├── utils.py                     # Wspólne narzędzia
└── README.md
```

## Trzy moduły

### Moduł 1: Generator (`run_generate.py`)
Generuje nowe mini-problemy z trzema typami rozwiązań.

```bash
# Domyślnie: 20 problemów, Llama 3 8B
python run_generate.py

# 50 problemów z Llama 3 70B
python run_generate.py model=llama3-70b n_problems=50

# Użyj odrzuconych przykładów (anti-overfitting)
python run_generate.py use_rejected=true
```

### Moduł 2: Solver (`run_solve.py`)
Rozwiązuje problemy (z Modułu 1) INNYM modelem — nie widzi oryginalnych rozwiązań.

```bash
# Rozwiąż problemy z generatora używając DeepSeek
python run_solve.py model=deepseek data.input_path=generated_problems.csv

# Generuj tylko kreatywne rozwiązania
python run_solve.py solve_mode=creative_only
```

### Moduł 3: Evaluator (`run_evaluate.py`)
Ocenia pary (problem, rozwiązanie) pod kątem feasibility i novelty.

```bash
# Oceń rozwiązane problemy
python run_evaluate.py data.input_path=solved_problems.csv

# Klasyfikuj rozwiązania (ordinary/creative/implausible)
python run_evaluate.py eval_mode=classify
```

## Typowy workflow

```bash
# Krok 1: Generuj problemy (Llama 3 70B)
python run_generate.py model=llama3-70b n_problems=30

# Krok 2: Rozwiąż te problemy innym modelem (DeepSeek)
python run_solve.py model=deepseek data.input_path=outputs/*/generated_problems.csv

# Krok 3: Oceń rozwiązania (Llama 3 70B jako sędzia)
python run_evaluate.py model=llama3-70b data.input_path=outputs/*/solved_problems.csv
```

## Hydra — krótkie wyjaśnienie

Hydra to system konfiguracji. Zamiast zmieniać kod, zmieniasz parametry z linii poleceń:

```bash
# Zmiana modelu:
python run_generate.py model=deepseek

# Zmiana wielu parametrów:
python run_generate.py model=llama3-70b n_problems=100 sampling_params.temperature=0.9

# Hydra tworzy folder outputs/ z timestampem dla każdego uruchomienia
# — znajdziesz tam logi i wyniki
```

## Modele

Pipeline jest zoptymalizowany pod **RTX 4090 (24GB VRAM)**. Dostępne modele:

**Tier 1 — łatwo mieści się na 4090:**
- `qwen2.5-7b` — najlepszy JSON output, domyślny model
- `llama3.1-8b` — silna baza odniesienia (wymaga licencji Meta na HF)
- `deepseek-r1-7b` — reasoning/chain-of-thought, dobry do ewaluacji
- `mistral-7b` — dobry do kreatywnej generacji
- `phi3.5-mini` — najmniejszy, najszybszy (~8GB VRAM)

**Tier 2 — mieści się, ale ciasno (zmniejszony context window):**
- `qwen2.5-14b` — najlepsza jakość w naszym zestawie
- `deepseek-r1-14b` — najlepszy ewaluator (chain-of-thought)
- `gemma2-9b` — Google, inna perspektywa (wymaga licencji Gemma)

```bash
# Pobieranie modeli:
python download_models.py --list                    # pokaż dostępne
python download_models.py --models qwen2.5-7b       # pobierz jeden
python download_models.py --tier1                    # pobierz Tier 1
python download_models.py --all                      # pobierz wszystkie (~142GB)
```

## Wymagania

```bash
pip install torch vllm hydra-core omegaconf wandb pandas tqdm huggingface_hub
huggingface-cli login  # potrzebne do gated models (Llama, Gemma)
```

## Format danych

Każdy mini-problem ma następujące pola:

| Pole | Opis | Wymaganie |
|------|-------|-----------|
| `problem` | Opis problemu (zaczyna się od "To") | 4-9 słów (bez "To") |
| `category` | Kategoria (np. visuo-spatial) | 1 z 8 kategorii |
| `constraint` | Ograniczenie w problemie | tekst |
| `ordinary_solution` | Rozwiązanie zwykłe | 1-3 słowa |
| `creative_solution` | Rozwiązanie kreatywne | 1-3 słowa |
| `implausible_solution` | Rozwiązanie bezsensowne | 1-3 słowa |
