import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

def parse_number(value):
    if value is None:
        return None

    cleaned = re.sub(r"[^\d.]", "", str(value))
    return float(cleaned) if cleaned else None

def parse_discount(value):
    if value is None:
        return None

    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


nltk.download('stopwords')

def flatten_product_details(details):
    if not details:
        return ''

    values = []
    for item in details:
        for key, value in item.items():
            if key != 'Style Code':
                values.append(str(value))

    return ' '.join(values)

class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text):
        # some docs may have no title/description
        if not isinstance(text, str) or not text:
            return []

        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        
        tokens = text.split()

        # we don't want 1-character tokens
        return [self.stemmer.stem(t) for t in tokens
                if t not in self.stop_words and len(t) > 1]

    def process_document(self, doc, output_fields):
        # keep only the fields we need to return in the results
        new_doc = {k: v for k, v in doc.items() if k in output_fields}

        new_doc['processed_title'] = self.preprocess_text(doc.get('title'))
        new_doc['processed_description'] = self.preprocess_text(doc.get('description'))
        new_doc["is_available"] = not doc.get("out_of_stock", False)
        new_doc["selling_price_value"] = parse_number(doc.get("selling_price"))
        new_doc["actual_price_value"] = parse_number(doc.get("actual_price"))
        new_doc["discount_percent"] = parse_discount(doc.get("discount"))
        new_doc["average_rating_value"] = parse_number(doc.get("average_rating"))
        # brand, sub_category and product details are joined as extra text for searching
        brand = doc.get('brand')
        sub_category = doc.get('sub_category')
        details = flatten_product_details(doc.get('product_details'))

        metadata = brand + ' ' + sub_category + ' ' + details
        new_doc['processed_metadata'] = self.preprocess_text(metadata)

        return new_doc

def load_and_preprocess_dataset(input_path, output_path, output_fields):
    preprocessor = TextPreprocessor()

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    processed = [preprocessor.process_document(d, output_fields) for d in data]

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
    return processed