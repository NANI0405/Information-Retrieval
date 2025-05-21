import json
import re
import glob
import time
import psutil
from collections import defaultdict


def load_data_in_blocks(file_path, block_size):
    
    print(f" Loading data from {file_path}...")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f" Error: File {file_path} not found. Please check if it exists.")
        return []
    
    print(f" Total documents found: {len(data)}")
    
    for i in range(0, len(data), block_size):
        print(f" Processing block {i // block_size + 1}...")
        yield data[i:i+block_size]  


def preprocess(text):
   
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    return text.split() 


def build_inverted_index(block, block_id):
    
    print(f" Building inverted index for block {block_id}...")
    index = defaultdict(set)
    
    for doc in block:
        doc_id = doc.get("Index", "UNKNOWN")
        words = preprocess(doc.get("Title", "") + " " + doc.get("Abstract", ""))
        
        for word in words:
            index[word].add(doc_id)

    block_filename = f"index_block_{block_id}.json"
    with open(block_filename, "w") as f:
        json.dump({k: list(v) for k, v in index.items()}, f)

    print(f" Block {block_id} saved as {block_filename}")
    return block_filename


def merge_indexes():
    
    print(" Merging all block indexes...")
    final_index = defaultdict(set)
    
    block_files = glob.glob("index_block_*.json")
    if not block_files:
        print(" No block index files found. Something went wrong.")
        return None

    for file in block_files:
        print(f" Merging {file}...")
        with open(file, "r") as f:
            block_index = json.load(f)
            for term, postings in block_index.items():
                final_index[term].update(postings)
    
    final_index_file = "final_inverted_index.json"
    with open(final_index_file, "w") as f:
        json.dump({k: list(v) for k, v in final_index.items()}, f)

    print(f" Final index saved in {final_index_file}")
    return final_index_file


def run_bsbi(file_path, block_size):
    
    print(f"\n Running BSBI with block_size = {block_size}...")

    block_files = []
    for block_id, block in enumerate(load_data_in_blocks(file_path, block_size)):
        if not block:
            print(f" Error: No documents found in block {block_id}.")
            return
        block_file = build_inverted_index(block, block_id)
        block_files.append(block_file)

    final_index_file = merge_indexes()
    if final_index_file:
        print(f" Final index created successfully: {final_index_file}")

#  Step 6: Benchmark Time & Memory Usage
def benchmark_bsbi(file_path, block_size):
    
    start_time = time.time()
    process = psutil.Process()

    run_bsbi(file_path, block_size)

    end_time = time.time()
    memory_used = process.memory_info().rss / (1024 * 1024)  

    print(f" Block Size: {block_size}, Time Taken: {end_time - start_time:.2f}s, Memory Used: {memory_used:.2f} MB\n")


if __name__ == "__main__":
    file_path = "bsbi_docs.json"  

    
    benchmark_bsbi(file_path, block_size=1)
    benchmark_bsbi(file_path, block_size=200000)