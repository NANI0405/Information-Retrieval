import os
import nltk
import pdfplumber
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text.lower())
    processed_words = [
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in words if word.isalnum() and word not in stop_words
    ]
    
    return " ".join(processed_words)

def extract_text_from_pdf(pdf_path):
    
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

ix = create_in("indexdir", schema)

def index_documents(pdf_files):
    
    writer = ix.writer()
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        processed_text = preprocess_text(text)
        writer.add_document(title=os.path.basename(pdf_file), content=processed_text)
    writer.commit()
    print("Indexing step is done")

def search_index(query_str):
    
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query, limit=5)  
        for result in results:
            print(f"Document: {result['title']}\n")


if __name__ == "__main__":
   
    pdf_files = ["d1.pdf","d2.pdf","d3.pdf","d4.pdf","d5.pdf","d6.pdf","d7.pdf"]  
    index_documents(pdf_files) 

    
    search_query = "Supreme Court ruling on criminal appeals"
    print("\n Search Results:")
    search_index(preprocess_text(search_query))
