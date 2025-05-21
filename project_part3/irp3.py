import os
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import nltk

nltk.download('punkt')

# --- Extract paragraphs from a single PDF ---
def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    raw_paragraphs = text.split('\n\n')
    paragraphs = [p.replace('\n', ' ').strip() for p in raw_paragraphs if len(p.strip()) > 40]
    return paragraphs

# --- Load all PDFs and extract paragraphs ---
def load_paragraphs_from_folder(folder_path):
    all_paragraphs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            paras = extract_paragraphs_from_pdf(path)
            all_paragraphs.extend(paras)
    return all_paragraphs

# --- Retrieve top N paragraphs using BM25 ---
def retrieve_with_bm25(paragraphs, query, top_n=5):
    tokenized_corpus = [word_tokenize(p.lower()) for p in paragraphs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = word_tokenize(query.lower())
    return bm25.get_top_n(tokenized_query, paragraphs, n=top_n)

# --- Score relevance using T5 ---
def rank_with_t5(paragraphs, query):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    scores = []
    for para in paragraphs:
        input_text = f"question: {query} context: {para}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output = model.generate(inputs, max_length=64, num_beams=2)
        generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Score = length of answer (simple proxy for now)
        scores.append((para, generated_answer, len(generated_answer)))

    # Sort by answer length (you can replace with something like BLEU later)
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[0]  # Most relevant paragraph

# === USAGE ===
folder_path = r"/Users/hariharaprasadgoud/developer/ir/Files"  # Your PDF folder path
query = "In terms of size, population and religious, cultural and linguistic diver-sity India resembles a continent much more than a single state."

# Step 1: Extract paragraphs
all_paragraphs = load_paragraphs_from_folder(folder_path)

# Step 2: Retrieve top N from BM25
bm25_top_paragraphs = retrieve_with_bm25(all_paragraphs, query, top_n=5)

# Step 3: Re-rank using T5
best_para, t5_answer, _ = rank_with_t5(bm25_top_paragraphs, query)

# === Output ===
print(f"\n🔍 Best Matching Paragraph:\n{best_para}")
