# IRWA_project

## Requirements
- Python 3.10 or later.

## Setup
From the project root, create and activate a virtual environment, then install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install nltk numpy pandas matplotlib seaborn wordcloud spacy jupyter
```

## Run
### Preprocess the data
From the project root, run:
```bash
python src/main.py
```
This reads `data/fashion_products_dataset.json` and creates `data/processed_fashion_products.json`.

### Run the EDA notebook
Start Jupyter from `src/` because the notebook uses a relative data path:
```bash
cd src
jupyter lab eda.ipynb
```
Alternatively, open `src/eda.ipynb` in VS Code and run the cells in order using `.venv` as the kernel. Run preprocessing first.
NLTK downloads its English stopwords automatically; the notebook downloads spaCy's `en_core_web_sm` model in its NER section. To install the model manually from the project root, run `python -m spacy download en_core_web_sm`.