# Part 1: Text Processing and Exploratory Data Analysis

## Contents
- `preprocess.py`: preprocesses product text by lowercasing, removing punctuation and English stop words, tokenizing, and applying Porter stemming. It creates processed title, description, and metadata fields, and parses availability, prices, discounts, and ratings.
- `eda.ipynb`: explores the processed dataset, text lengths and vocabulary, popular sub-categories, brands and sellers, commercial attributes, a description word cloud, and named entities from a sample of 500 descriptions.
- `../../config.yaml`: defines the dataset paths and the source fields retained during preprocessing.

## Reproduce the results
1. Place `fashion_products_dataset.json` in the repository's `data/` directory.
2. From the repository root, install the dependencies if needed:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `eda.ipynb` and run it with `project_progress/part_1` as the working directory. Its first cell creates `data/processed_fashion_products.json` if it does not already exist. The NER section downloads the spaCy model `en_core_web_sm` and analyzes a random sample of 500 descriptions.

