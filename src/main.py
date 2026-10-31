import json
from pathlib import Path

import yaml

from preprocess import load_and_preprocess_dataset, TextPreprocessor
from indexing import build_tfidf_index, search_tfidf

ROOT_DIR = Path(__file__).parents[1]
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

    # ---- part 2: indexing ----
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


if __name__ == '__main__':
    main()
