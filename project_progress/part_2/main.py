import sys
import json
from pathlib import Path
import pandas as pd
import yaml

# root of the repository (project_progress/part_2 -> ../..), so we can import myapp
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from project_progress.part_1.preprocess import TextPreprocessor, load_and_preprocess_dataset
from indexing import build_tfidf_index,search_tfidf
from evaluation import evaluate_system
CONFIG_PATH = ROOT_DIR / 'config.yaml'


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    config = load_config(CONFIG_PATH)

    data_dir = ROOT_DIR / config['paths']['data_dir']
    input_path = data_dir / config['paths']['raw_dataset']
    output_path = data_dir / config['paths']['processed_dataset']

    output_fields = config['preprocessing']['output_fields']
    text_fields = config['indexing']['text_fields']
    top_k = config['search']['top_k']
    test_queries = config['search']['test_queries']

    # used to preprocess the queries in the same way as the documents
    preprocessor = TextPreprocessor()

    # ---- part 1: preprocessing (only if we don't have the processed file yet) ----
    if output_path.exists():
        with open(output_path, encoding='utf-8') as f:
            docs = json.load(f)
    else:
        docs = load_and_preprocess_dataset(input_path, output_path, output_fields=output_fields)

    # ---- part 2.1: indexing ----
    index, tf, df, idf, pid_info = build_tfidf_index(docs, text_fields)

    print('Number of documents:', len(docs))
    print('Number of terms in the index:', len(index))

    # ---- search + ranking of our test queries ----
    for query in test_queries:
        ranked = search_tfidf(query, preprocessor, index, tf, idf)

        print('Query:', query)
        print('Documents containing all the terms:', len(ranked))

        # print only the top k
        for position in range(min(top_k, len(ranked))):
            score = ranked[position][0]
            pid = ranked[position][1]
            title = pid_info[pid]['title']
            print(position + 1, round(score, 3), pid, title)

    # ---- part 2.2: evaluation ---- 
    labels_path = data_dir / config['paths']['validation_labels']
    val_df = pd.read_csv(labels_path)

    val_queries = config['evaluation']['validation_queries']

    results = evaluate_system(
        queries=val_queries,
        labels_df=val_df,
        index=index,
        tf=tf,
        idf=idf,
        preprocessor=preprocessor,
        k=top_k
    )

    for q_idx, query_text, metrics in results['query_metrics']:
        print(f"* Query {q_idx}: {query_text}")
        print(f"P@{top_k}: {metrics['P@K']:.3f}")
        print(f"R@{top_k}: {metrics['R@K']:.3f}")
        print(f"F1@{top_k}: {metrics['F1@K']:.3f}")
        print(f"AP@{top_k}: {metrics['AP@K']:.3f}")
        print(f"RR@{top_k}: {metrics['RR@K']:.3f}")
        print(f"NDCG@{top_k}: {metrics['NDCG@K']:.3f}\n")

    print(f"MAP: {results['MAP']:.3f}")
    print(f"MRR: {results['MRR']:.3f}")
    
    # ---- part 2.3: evaluation own search queries---- 
    print(f"EVALUATION Test Queries")

    labels_path = data_dir / config['paths']['validation2_labels']
    val_df = pd.read_csv(labels_path)

    val_queries = config['search']['test_queries']

    results = evaluate_system(
        queries=val_queries,
        labels_df=val_df,
        index=index,
        tf=tf,
        idf=idf,
        preprocessor=preprocessor,
        k=top_k
    )

    for q_idx, query_text, metrics in results['query_metrics']:
        print(f"* Query {q_idx}: {query_text}")
        print(f"P@{top_k}: {metrics['P@K']:.3f}")
        print(f"R@{top_k}: {metrics['R@K']:.3f}")
        print(f"F1@{top_k}: {metrics['F1@K']:.3f}")
        print(f"AP@{top_k}: {metrics['AP@K']:.3f}")
        print(f"RR@{top_k}: {metrics['RR@K']:.3f}")
        print(f"NDCG@{top_k}: {metrics['NDCG@K']:.3f}\n")

    print(f"MAP: {results['MAP']:.3f}")
    print(f"MRR: {results['MRR']:.3f}")

if __name__ == '__main__':
    main()
