import os
import re
import fitz  # PyMuPDF
import torch
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from termcolor import colored
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')
nltk.download('stopwords')

# === Config ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

def extract_paragraphs_from_pdf(pdf_path):
    print(f"Extracting from PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    sentences = sent_tokenize(full_text)
    paragraphs, chunk = [], []
    for sentence in sentences:
        chunk.append(sentence.strip())
        if len(chunk) >= 3:
            paragraphs.append(" ".join(chunk))
            chunk = []
    if chunk:
        paragraphs.append(" ".join(chunk))
    valid_paragraphs = [p for p in paragraphs if len(p) > 40]
    print(f"Extracted {len(valid_paragraphs)} valid paragraphs.")
    return valid_paragraphs

def extract_all_paragraphs_from_folder(folder_path):
    all_paragraphs, sources = [], []
    print(f"Reading PDFs from folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            paragraphs = extract_paragraphs_from_pdf(path)
            all_paragraphs.extend(paragraphs)
            sources.extend([filename] * len(paragraphs))
    print(f"Total paragraphs extracted: {len(all_paragraphs)}")
    return all_paragraphs, sources

def bm25_retrieve(paragraphs, query, top_n=3, verbose=True):
    tokenized_paragraphs = [preprocess_text(p) for p in paragraphs]
    bm25 = BM25Okapi(tokenized_paragraphs)
    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    top_paras = [paragraphs[i] for i in top_indices]
    if verbose:
        print("\n\033[1;35mBM25 Scores and Highlighted Paragraphs:\033[0m")
        for i in top_indices:
            print(f"\n\033[1;32mScore: {scores[i]:.4f}\033[0m")
            print(highlight_query_terms(paragraphs[i], query))
    return top_paras

def highlight_query_terms(text, query):
    for word in set(word_tokenize(query.lower())):
        pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
        text = pattern.sub(colored(r"\1", "yellow", attrs=["bold"]), text)
    return text

def generate_answer_with_t5(paragraphs, query):
    context = " ".join(paragraphs)
    input_text = f"answer the question: {query} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_answer(generated, reference):
    bleu = sentence_bleu([reference.split()], generated.split())
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(reference, generated)
    P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)
    print("\n\033[1;35mEvaluation Metrics:\033[0m")
    print(f"BLEU Score: {bleu:.4f}")
    print(f"ROUGE-L F1 Score: {rouge['rougeL'].fmeasure:.4f}")
    print(f"BERTScore F1: {F1.item():.4f}")

def process_query(query, reference_answer, paragraphs):
    print(f"\nProcessing: {query}")
    top_paragraphs = bm25_retrieve(paragraphs, query, top_n=3)
    if not top_paragraphs:
        print("No relevant paragraphs found.")
        return
    generated = generate_answer_with_t5(top_paragraphs, query)
    print("\n\033[1;33mGenerated Answer:\033[0m \033[1;36m" + generated + "\033[0m")
    evaluate_answer(generated, reference_answer)

# === Entry Point ===
if __name__ == "_main_":
    # Provide fallback paths here if you want to hardcode dataset locations
    default_pdf_folder = r"D:\\IRP Dataset"
    default_query_file = r"D:\\project\\queries.txt"

    print("Enter the full path to your PDF folder (or press Enter to use default):")
    folder_path = input(f"PDF Folder Path [Default: {default_pdf_folder}]: ").strip()
    if not folder_path:
        folder_path = default_pdf_folder

    print("Enter the full path to your query file (or press Enter to use default):")
    query_file = input(f"Query File Path [Default: {default_query_file}]: ").strip()
    if not query_file:
        query_file = default_query_file

    if not os.path.exists(folder_path):
        print("❌ PDF folder path does not exist.")
        exit()
    if not os.path.exists(query_file):
        print("❌ Query file path does not exist.")
        exit()

    print("✅ Paths accepted. Starting extraction and evaluation...")
    paragraphs, _ = extract_all_paragraphs_from_folder(folder_path)

    with open(query_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            print("❌ Query file is empty.")
        for line in lines:
            if '|||' not in line:
                print(f"⚠ Skipping invalid line: {line.strip()}")
                continue
            query, reference = map(str.strip, line.split('|||'))
            print("\n\033[1;34mProcessing Query:\033[0m", query)
            process_query(query, reference, paragraphs)