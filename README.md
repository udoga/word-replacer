# Word Replacer

### Setup

* Initialize virtual env: `python3 -m venv .venv`
* Activate virtual env: `source .venv/bin/activate` or `.venv\Scripts\activate`
* Install dependencies: `pip install -r requirements.txt`
* Download wordnet: `python -c "import nltk; nltk.download('wordnet')"`
* Download dataset: `./download_dataset.sh`
* Run unit tests: `pytest`
* Run substituter or benchmark: `python main.py`
* For running Llama: `huggingface-cli login`

### Results

| Dataset | Method  | best  | best-mode | oot   | oot-mode | P@1   |
|---------|---------|-------|-----------|-------|----------|-------|
| LS07    | Dropout | 22.24 | 36.11     | 57.51 | 70.73    | 50.88 |
| LS07    | Concat  | 21.34 | 35.76     | 56.29 | 70.50    | 47.12 |
| LS07    | Pattern | 21.66 | 34.68     | 56.22 | 70.77    | 49.19 |
