# Part 2: Indexing and Evaluation

## Contents
- `indexing.py`: builds the inverted index, performs conjunctive (AND) search, calculates term statistics, and ranks matching documents with TF-IDF. Document term frequency uses sublinear weighting and normalization; inverse document frequency uses base-2 logarithms.
- `evaluation.py`: calculates Precision@K, Recall@K, F1@K, AP@K, RR@K, NDCG@K, MAP, and MRR.
- `main.py`: loads the configuration and data, builds the index, prints results for the configured test queries, and evaluates both query sets against their relevance labels.
- `report_enrichment.ipynb`: generates index statistics and report tables/figures, including indexing times, field and vocabulary statistics, a preprocessing example, and query-term analysis.
- `../../config.yaml`: defines the data paths, indexed text fields, test and validation queries, and K.

## Reproduce the results
Python version used for verification: **3.14**.

Install the project dependencies from the repository root if they are not installed yet:

```bash
pip install -r requirements.txt
```


From the repository root, run the Part 2 pipeline:

```bash
python project_progress/part_2/main.py
```



