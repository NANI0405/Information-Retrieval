import nltk
nltk.download('punkt_tab')
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
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    raw_paragraphs = full_text.split('\n\n')
    paragraphs = [para.strip().replace('\n', ' ') for para in raw_paragraphs if len(para.strip()) > 40]
    return paragraphs

# --- Extract all paragraphs from a folder of PDFs ---
def extract_all_paragraphs_from_folder(folder_path):
    all_paragraphs = []
    sources = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            paragraphs = extract_paragraphs_from_pdf(path)
            all_paragraphs.extend(paragraphs)
            sources.extend([filename] * len(paragraphs))
    return all_paragraphs, sources

# --- BM25 Retrieval ---
def bm25_retrieve(paragraphs, query, top_n=3):
    tokenized_paragraphs = [word_tokenize(p.lower()) for p in paragraphs]
    bm25 = BM25Okapi(tokenized_paragraphs)
    tokenized_query = word_tokenize(query.lower())
    top_paras = bm25.get_top_n(tokenized_query, paragraphs, n=top_n)
    return top_paras

# --- T5 Answer Generation ---
def generate_answer_with_t5(paragraphs, query):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    context = " ".join(paragraphs)
    input_text = f"answer the question: {query} context: {context}"

    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === MAIN PIPELINE ===
folder_path = "D:\project"  # <-- Replace this with your folder path
query = "When the Judge ordered the victims husbands to pay maintenance, they stayed their cases through their lawyers"

# Step 1: Extract from all PDFs
paragraphs, sources = extract_all_paragraphs_from_folder(folder_path)

# Step 2: Retrieve relevant paragraphs
top_paragraphs = bm25_retrieve(paragraphs, query, top_n=3)

# Step 3: Generate answer
answer = generate_answer_with_t5(top_paragraphs, query)

# Step 4: Show results
print("\n Top Relevant Paragraphs:")
for i, para in enumerate(top_paragraphs):
    print(f"\n{i+1}. {para}")

print(f"\n Answer: {answer}")
