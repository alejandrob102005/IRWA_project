import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

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

    def process_document(self, doc):
        # copy so we don't modify the original dict
        new_doc = doc.copy()
        #only title and description are text fields we need 
        new_doc['processed_title'] = self.preprocess_text(doc.get('title'))
        new_doc['processed_description'] = self.preprocess_text(doc.get('description'))
        return new_doc


def load_and_preprocess_dataset(input_path, output_path):
    preprocessor = TextPreprocessor()

    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    processed = [preprocessor.process_document(d) for d in data]

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
    return processed