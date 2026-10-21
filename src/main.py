from pathlib import Path

from preprocess import load_and_preprocess_dataset

DATA_DIR = Path(__file__).parents[1] / 'Data'
INPUT_PATH = DATA_DIR / 'fashion_products_dataset.json'
OUTPUT_PATH = DATA_DIR / 'processed_fashion_products.json'

def main():
    OUTPUT_FIELDS = ['pid', 'title', 'description', 'brand', 'category', 'sub_category',
                 'product_details', 'seller', 'out_of_stock', 'selling_price',
                 'discount', 'actual_price', 'average_rating', 'url']

    load_and_preprocess_dataset(INPUT_PATH, OUTPUT_PATH, output_fields=OUTPUT_FIELDS)

if __name__ == '__main__':
    main()