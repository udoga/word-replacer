# Word Replacer

* Initialize virtual env: `python3 -m venv .venv`
* Activate virtual env: `source .venv/bin/activate` or `.venv\Scripts\activate`
* Install dependencies: `pip install -r requirements.txt`
* Download wordnet: `python -c "import nltk; nltk.download('wordnet')"`
* Download dataset: `./download_dataset.sh`
* Run unit tests: `pytest`
* Run substituter or benchmark: `python main.py`
