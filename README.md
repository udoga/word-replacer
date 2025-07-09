# Word Replacer

## Setup
```
python3 -m venv .venv
source .venv/bin/activate # .venv\Scripts\activate for Windows
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
./download_dataset.sh
pytest
python main.py
```
