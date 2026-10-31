import math
from collections import defaultdict, Counter

import numpy as np


# text_fields comes from config.yaml (indexing -> text_fields)
def get_doc_terms(doc, text_fields):
    terms = []
    for field in text_fields:
        tokens = doc.get(field)
        # some documents have empty fields, we just skip them
        if not tokens:
            continue
        for token in tokens:
            terms.append(token)
    return terms


def build_inverted_index(docs, text_fields):
    
    index = defaultdict(list)
    product_id_info = {}
    for doc in docs:
        product_id = doc['pid']
        product_id_info[product_id] = doc
        terms = get_doc_terms(doc, text_fields)
        # we use a set so each document is added only once to the posting list
        # of a term, even if the term appears several times in the document
        unique_terms = set(terms)
        for term in unique_terms:
            index[term].append(product_id)

    return index, product_id_info


def conjunctive_search(query_terms, index):
    # empty query (for example only stopwords) -> no results
    if len(query_terms) == 0:
        return set()

    result = None
    for term in query_terms:
        # if one of the terms does not appear in the collection, no document can
        # contain all the terms, so the answer is empty
        if term not in index:
            return set()
        # documents that contain this term
        term_docs = set(index[term])

        # intersection with the documents of the previous terms
        if result is None:
            result = term_docs
        else:
            result = result.intersection(term_docs)

        # if it is already empty we can stop as querys will be AND queries
        if len(result) == 0:
            break

    return result



def get_term_statistics(docs, index, text_fields):
    
    N = len(docs)
    # total frequency of each term in the collection
    tf_total = Counter()
    for doc in docs:
        terms = get_doc_terms(doc, text_fields)
        for term in terms:
            tf_total[term] += 1

    stats = []
    for term in index:
        df = len(index[term])
        idf = math.log(N / df)
        stats.append({
            'term': term,
            'tf_total': tf_total[term],
            'df': df,
            'idf': round(idf, 3),
            # collection-level tf-idf, it is only a heuristic to find terms that are
            # frequent but not present in every document
            'tf_idf': round(tf_total[term] * idf, 2),
        })

    return stats


def build_tfidf_index(docs, text_fields):
    index = defaultdict(list)
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = {}
    product_id_info = {}

    N = len(docs)

    for doc in docs:
        product_id = doc['pid']
        product_id_info[product_id] = doc

        terms = get_doc_terms(doc, text_fields)

        # count how many times each term appears in this document
        term_counts = Counter(terms)

        # we calculate sublinear TF: tf = 1 + log2(f)
        tf_sublinear = {}
        for term, count in term_counts.items():
            tf_sublinear[term] = 1 + math.log2(count)

        # we calculate document norm
        norm = math.sqrt(sum(w ** 2 for w in tf_sublinear.values()))

        for term, sublinear_weight in tf_sublinear.items():
            index[term].append(product_id)
            tf[term].append(np.round(sublinear_weight / norm, 4))
            df[term] += 1

    # we calculate base-2 IDF
    for term in df:
        idf[term] = np.round(math.log2(N / df[term]), 4)

    return index, tf, df, idf, product_id_info


def rank_documents(query_terms, docs, index, tf, idf):
    # we use the unique terms of the query, each one is a dimension of the vectors
    unique_terms = []
    for term in query_terms:
        if term not in unique_terms:
            unique_terms.append(term)

    # query vector -> tf of the term in the query * idf
    query_counts = Counter(query_terms)
    query_weights = {term: 1 + math.log2(count) for term, count in query_counts.items()}
    query_norm = math.sqrt(sum(w ** 2 for w in query_weights.values()))

    query_vector = [0] * len(unique_terms)
    # each document has a vector with one position per query term, we do this as in the lab
    doc_vectors = defaultdict(lambda: [0] * len(unique_terms))

    for i in range(len(unique_terms)):
        term = unique_terms[i]
        if term not in index:
            continue

        query_vector[i] = (query_weights[term] / query_norm) * idf[term]

        # go through the posting list of the term and fill the vectors of the
        # documents that we want to rank
        # doc_position is the position of the document in the posting list,
        # we need it because tf[term] is stored in the same order
        postings = index[term]
        for doc_position in range(len(postings)):
            product_id = postings[doc_position]
            if product_id in docs:
                doc_vectors[product_id][i] = tf[term][doc_position] * idf[term]

    # score of each document = dot product with the query vector
    # (the tf of the documents is already normalized)
    doc_scores = []
    for product_id in doc_vectors:
        score = np.dot(doc_vectors[product_id], query_vector)
        doc_scores.append([score, product_id])

    doc_scores.sort(reverse=True)

    return doc_scores


def search_tfidf(query, preprocessor, index, tf, idf):
    
    query_terms = preprocessor.preprocess_text(query)
    docs = conjunctive_search(query_terms, index)
    ranked_docs = rank_documents(query_terms, docs, index, tf, idf)
    return ranked_docs