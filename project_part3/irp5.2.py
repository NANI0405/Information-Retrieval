import os
import nltk
import fitz  
import torch
import re
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from termcolor import colored

nltk.download('punkt')


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)


def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    sentences = sent_tokenize(full_text)
    paragraphs = []
    chunk = []
    for sentence in sentences:
        chunk.append(sentence.strip())
        if len(chunk) >= 3:
            paragraphs.append(" ".join(chunk))
            chunk = []
    if chunk:
        paragraphs.append(" ".join(chunk))
    return [p for p in paragraphs if len(p) > 40]

def extract_all_paragraphs_from_folder(folder_path):
    all_paragraphs, sources = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            paragraphs = extract_paragraphs_from_pdf(path)
            all_paragraphs.extend(paragraphs)
            sources.extend([filename] * len(paragraphs))
    return all_paragraphs, sources

def bm25_retrieve(paragraphs, query, top_n=3, verbose=True):
    tokenized_paragraphs = [list(set(word_tokenize(p.lower()))) for p in paragraphs]
    bm25 = BM25Okapi(tokenized_paragraphs)
    tokenized_query = word_tokenize(query.lower())
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
    """Highlight query words/phrases inside a paragraph using termcolor."""
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

def highlight_answer_in_paragraph(paragraph, answer):
    
    pattern = re.escape(answer.strip())
    highlighted = re.sub(pattern, f"\033[1;32m{answer}\033[0m", paragraph, flags=re.IGNORECASE)
    return highlighted

def calculate_accuracy_per_paragraph(answer, paragraphs):
    answer_tokens = set(word_tokenize(answer.lower()))
    results = []
    for para in paragraphs:
        para_tokens = set(word_tokenize(para.lower()))
        matched_tokens = answer_tokens & para_tokens
        if not answer_tokens:
            accuracy = 0.0
        else:
            accuracy = len(matched_tokens) / len(answer_tokens) * 100
        results.append(round(accuracy, 2))
    return results



folder_path = "D:\poj"
query = input("\nEnter your query: ")

paragraphs, sources = extract_all_paragraphs_from_folder(folder_path)
top_paragraphs = bm25_retrieve(paragraphs, query, top_n=3)
answer = generate_answer_with_t5(top_paragraphs, query)


accuracies = calculate_accuracy_per_paragraph(answer, top_paragraphs)

print("\n\033[1;34mTop Relevant Paragraphs and Accuracy:\033[0m")
for i, (para, acc) in enumerate(zip(top_paragraphs, accuracies)):
    highlighted = highlight_answer_in_paragraph(para, answer)
    print(f"\n{i+1}. Accuracy: \033[1;36m{acc:.2f}%\033[0m")
    print(highlighted)

print(f"\n\033[1;33mGenerated Answer:\033[0m \033[1;36m{answer}\033[0m\n")




