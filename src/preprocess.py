import json
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text: str) -> list:
        
        if not text or not isinstance(text, str):
            return []

        # Lowercasing
        text = text.lower()

        # Replace punctuation with spaces and clean non-alphanumeric characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stop words and short tokens, and apply stemming
        cleaned_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]

        return cleaned_tokens

    def process_document(self, doc: dict) -> dict:
        
        processed_doc = doc.copy()
        
        # We only preprocess the text fields: title and description
        processed_doc['processed_title'] = self.preprocess_text(doc.get('title', ''))
        processed_doc['processed_description'] = self.preprocess_text(doc.get('description', ''))
        
        return processed_doc



def load_and_preprocess_dataset(input_filepath: str, output_filepath: str = None):
    
    preprocessor = TextPreprocessor()

    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = [preprocessor.process_document(doc) for doc in data]

    if output_filepath:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"Preprocessed dataset successfully saved to: {output_filepath}")

    return processed_data

if __name__ == "__main__":
    # default file paths
    input_file = "Data/fashion_products_dataset.json"
    output_file = "Data/processed_fashion_products.json"
    
    processed_dataset = load_and_preprocess_dataset(input_file, output_file)
    print(f"Successfully processed {len(processed_dataset)} documents.")
    