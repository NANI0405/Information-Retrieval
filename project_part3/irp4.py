import os
import nltk
import fitz  # PyMuPDF
import torch
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Download necessary tokenizer
nltk.download('punkt')


# --- Paragraph Extraction ---
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
        paragraphs.append(" ".join(chunk))  # leftover

    return [p for p in paragraphs if len(p) > 40]


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

    print("\nRetrieved Top Paragraphs with Scores:")
    for i in top_indices:
        print(f"\nScore: {scores[i]:.4f}\n{paragraphs[i]}")
    
    return top_paras


# --- T5 QA ---
def generate_answer_with_t5(paragraphs, query):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    context = " ".join(paragraphs)
    input_text = f"question: {query} \n\ncontext: {context} \n\nanswer:"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    outputs = model.generate(
        inputs, max_length=200, num_beams=4, early_stopping=True, num_return_sequences=1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- Highlight Matching Answer ---
def highlight_answer_in_paragraphs(paragraphs, answer):
    highlighted = []
    for p in paragraphs:
        lower_p = p.lower()
        lower_ans = answer.lower()
        if lower_ans in lower_p:
            start = lower_p.find(lower_ans)
            end = start + len(answer)
            highlighted_p = p[:start] + ">>>" + p[start:end] + "<<<" + p[end:]
            highlighted.append(highlighted_p)
        else:
            highlighted.append(p)
    return highlighted


# === MAIN PIPELINE ===
folder_path = "/Users/hariharaprasadgoud/developer/ir/Files"
query = input("Enter your query: ")

# Step 1: Extract from all PDFs
paragraphs, sources = extract_all_paragraphs_from_folder(folder_path)

# Step 2: Retrieve relevant paragraphs
top_paragraphs = bm25_retrieve(paragraphs, query, top_n=3)

# Step 3: Generate answer
answer = generate_answer_with_t5(top_paragraphs, query)

# Step 4: Fallback if empty or useless
if not answer or answer.lower() in ["none", ""]:
    print("\n⚠️  T5 did not produce a clear answer, showing top result as fallback.\n")
    answer = top_paragraphs[0]

# Step 5: Highlight answer in context
highlighted = highlight_answer_in_paragraphs(top_paragraphs, answer)

# Step 6: Print Results
print("\n🔍 Top Relevant Paragraphs with Highlighted Answer:")
for i, para in enumerate(highlighted):
    print(f"\n{i+1}. {para}")

print(f"\n✅ Final Answer: {answer}")
