import json
from pathlib import Path
import pandas as pd
import yaml

from evaluation import evaluate_query, mean_average_precision, mean_reciprocal_rank
from indexing import build_tfidf_index, search_tfidf
from preprocess import TextPreprocessor, load_and_preprocess_dataset

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / 'config.yaml'


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_validation():
    config = load_config(CONFIG_PATH)
    data_dir = ROOT_DIR / config['paths']['data_dir']

    processed_path = data_dir / config['paths']['processed_dataset']
    raw_path = data_dir / config['paths']['raw_dataset']

    if processed_path.exists():
        with open(processed_path, encoding='utf-8') as f:
            docs = json.load(f)
    else:
        docs = load_and_preprocess_dataset(
            raw_path,
            processed_path,
            output_fields=config['preprocessing']['output_fields'],
        )

    text_fields = config['indexing']['text_fields']
    index, tf, df, idf, pid_info = build_tfidf_index(docs, text_fields)
    preprocessor = TextPreprocessor()

    labels_path = data_dir / config['paths']['validation_labels']
    val_df = pd.read_csv(labels_path)

    val_queries = config['evaluation']['validation_queries']
    k_cutoff = config['search']['top_k']

    ap_list = []
    rr_list = []

    for q_idx, query in enumerate(val_queries, start=1):
        ranked_results = search_tfidf(query, preprocessor, index, tf, idf)

        query_labels = val_df[val_df['query'] == query]
        rel_map = dict(zip(query_labels['pid'], query_labels['label']))
        total_relevant = int(
            query_labels[query_labels['label'] == 1].shape[0]
        )

        retrieved_relevance = [
            rel_map.get(pid, 0) for score, pid in ranked_results
        ]

        metrics = evaluate_query(
            retrieved_relevance, total_relevant, k=k_cutoff
        )

        ap_list.append(metrics['AP@K'])
        rr_list.append(metrics['RR@K'])

        print(f'a. Query {q_idx}: {query}')
        print(f"P@{k_cutoff}: {metrics['P@K']:.3f}")
        print(f"R@{k_cutoff}: {metrics['R@K']:.3f}")
        print(f"F1@{k_cutoff}: {metrics['F1@K']:.3f}")
        print(f"AP@{k_cutoff}: {metrics['AP@K']:.3f}")
        print(f"RR@{k_cutoff}: {metrics['RR@K']:.3f}")
        print(f"NDCG@{k_cutoff}: {metrics['NDCG@K']:.3f}\n")

    map_score = mean_average_precision(ap_list)
    mrr_score = mean_reciprocal_rank(rr_list)

    print(f'MAP: {map_score:.3f}')
    print(f'MRR: {mrr_score:.3f}')


if __name__ == '__main__':
    run_validation()