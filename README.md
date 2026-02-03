# Word Replacer

### Setup

* Initialize virtual env: `python3 -m venv .venv`
* Activate virtual env: `source .venv/bin/activate` or `.venv\Scripts\activate`
* Install dependencies: `pip install -r requirements.txt`
* Download NLTK resources: `python -m nltk.downloader wordnet stopwords`
* Download dataset: `./download_dataset.sh`
* Run unit tests: `pytest`
* For Llama (optional): `hf auth login --token YOUR_HF_TOKEN`

### Examples

* Print usage: `./word_replacer.py`
* Substitute demo example with dropout method: `./word_replacer.py demo dropout`
* Substitute user input: `./word_replacer.py substitute dropout "The roses are bright" bright 3`
* Benchmark on lst_trial dataset: `./word_replacer.py benchmark dropout lst_trial report.txt`

### Benchmarks

| Dataset   | Method     | best  | best-mode | oot   | oot-mode | P@1   |
|-----------|------------|-------|-----------|-------|----------|-------|
| LS07-test | Zhou       | 18.77 | 29.52     | 54.68 | 64.96    | 43.29 |
| LS07-test | Dropout    | 22.24 | 36.11     | 57.51 | 70.63    | 50.88 |
| LS07-test | Concat     | 21.30 | 35.69     | 56.23 | 70.46    | 47.04 |
| LS07-test | Pattern    | 21.66 | 34.68     | 56.22 | 70.77    | 49.19 |
| LS07-test | GPT2-Beam  | 17.84 | 28.93     | 51.17 | 66.89    | 41.56 |
| LS07-test | GPT4-List  | 30.08 | 48.27     | 71.23 | 83.25    | 68.60 |
| LS07-test | Llama-List | 24.88 | 40.32     | 59.54 | 72.37    | 57.79 |

### Dissertation

Full research methodology and benchmarking results are available in my [dissertation](dissertation.pdf).
