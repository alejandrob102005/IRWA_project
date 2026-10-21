from pathlib import Path

from preprocess import load_and_preprocess_dataset

DATA_DIR = Path(__file__).parents[1] / 'Data'
INPUT_PATH = DATA_DIR / 'fashion_products_dataset.json'
OUTPUT_PATH = DATA_DIR / 'processed_fashion_products.json'

def main():
    load_and_preprocess_dataset(INPUT_PATH, OUTPUT_PATH)
    

if __name__ == '__main__':
    main()