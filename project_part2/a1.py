import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


with open("bool_docs.json", "r") as f:
    documents = json.load(f)


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess(text):
    tokens = word_tokenize(text.lower()) 
    tokens = [word for word in tokens if word.isalnum()]  
    tokens = [word for word in tokens if word not in stop_words]  
    tokens = [stemmer.stem(word) for word in tokens]  
    return tokens


preprocessed_docs = {}
for doc in documents:
    doc_id = doc["Index"]
    text = f"{doc['Title']} {doc['Abstract']}"
    preprocessed_docs[doc_id] = preprocess(text)


from collections import defaultdict


inverted_index = defaultdict(set)

for doc_id, terms in preprocessed_docs.items():
    for term in terms:
        inverted_index[term].add(doc_id)


inverted_index = {term: list(docs) for term, docs in inverted_index.items()}



def boolean_retrieval(query, inverted_index, total_docs):
    terms = query.split()
    result_set = set()

    operator = None
    current_set = set()
    
    for term in terms:
        if term in {"AND", "OR", "NOT"}:
            operator = term
        else:
            stemmed_term = stemmer.stem(term.lower())  
            term_docs = set(inverted_index.get(stemmed_term, []))  
            
            if operator is None:
                current_set = term_docs
            elif operator == "AND":
                current_set &= term_docs
            elif operator == "OR":
                current_set |= term_docs
            elif operator == "NOT":
                current_set -= term_docs
    
    return sorted(list(current_set)) 


total_docs = set(preprocessed_docs.keys())



with open("bool_queries.json", "r") as f:
    queries = json.load(f)


results = {}
for q in queries:
    query_text = q["query"]
    results[query_text] = boolean_retrieval(query_text, inverted_index, total_docs)

print(results)  





    
