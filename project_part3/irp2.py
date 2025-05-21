import os
import nltk
import fitz  # PyMuPDF
import torch
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
from colorama import init, Fore, Style
import re
init(autoreset=True)

# Download required NLTK tokenizer
nltk.download('punkt')  

# --- Extract paragraphs from a single PDF ---
def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    sentences = sent_tokenize(full_text)
    # Group every 2–3 sentences into a paragraph
    paragraphs = []
    chunk = []
    for sentence in sentences:
        chunk.append(sentence.strip())
        if len(chunk) >= 3:
            paragraphs.append(" ".join(chunk))
            chunk = []
    if chunk:
        paragraphs.append(" ".join(chunk))  # leftover

    return [p for p in paragraphs if len(p) > 40]

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
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    top_paras = [paragraphs[i] for i in top_indices]

    # Optional: print scores for debugging
    for i in top_indices:
        print(f"\nScore: {scores[i]:.4f}\n{paragraphs[i]}")
    
    return top_paras

# --- T5 Answer Generation ---
def generate_answer_with_t5(paragraphs, query):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    best_answer = ""
    best_length = 0

    for para in paragraphs:
        input_text = f"answer the question: {query} context: {para}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if len(answer.split()) > best_length:
            best_answer = answer
            best_length = len(answer.split())

def highlight_answer_in_paragraphs(paragraphs, answer):
    highlighted = []
    answer_escaped = re.escape(answer.strip())
    pattern = re.compile(answer_escaped, re.IGNORECASE)

    for para in paragraphs:
        match = pattern.search(para)
        if match:
            start, end = match.span()
            highlighted_para = (
                para[:start] +
                Fore.GREEN + Style.BRIGHT + para[start:end] + Style.RESET_ALL +
                para[end:]
            )
            highlighted.append(highlighted_para)
        else:
            highlighted.append(para)
    return highlighted


# === MAIN PIPELINE ===
folder_path = "/Users/hariharaprasadgoud/developer/ir/Files"  # <-- your folder path
query = input("Enter your query: ")

# Step 1: Extract from all PDFs
paragraphs, sources = extract_all_paragraphs_from_folder(folder_path)

# Step 2: Retrieve relevant paragraphs
top_paragraphs = bm25_retrieve(paragraphs, query, top_n=3)

# Step 3: Generate answer
answer = generate_answer_with_t5(top_paragraphs, query)

highlighted = highlight_answer_in_paragraphs(top_paragraphs, answer)


# Step 4: Show results
print(Fore.YELLOW + "\n🔍 Top Relevant Paragraphs with Highlighted Answer:" + Style.RESET_ALL)
for i, para in enumerate(highlighted):
    para_colored = para.replace(">>>", Fore.GREEN + Style.BRIGHT).replace("<<<", Style.RESET_ALL)
    print(f"\n{Fore.CYAN}{i+1}.{Style.RESET_ALL} {para_colored}")

print(Fore.MAGENTA + f"\n✅ Final Answer: {Fore.GREEN}{answer}" + Style.RESET_ALL)


