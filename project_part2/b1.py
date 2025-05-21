import json
from collections import Counter
from nltk.util import ngrams


def load_dictionary(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def get_ngrams(word, n=3):
    return set(ngrams(word, n))

def correct_word_ngram(word, dictionary):
    word_ngrams = get_ngrams(word, 3) | get_ngrams(word, 2)
    best_match, best_score = None, 0

    for dict_word in dictionary:
        dict_ngrams = get_ngrams(dict_word, 3) | get_ngrams(dict_word, 2)
        score = len(word_ngrams & dict_ngrams) / len(word_ngrams | dict_ngrams)
        if score > best_score:
            best_match, best_score = dict_word, score

    return best_match if best_match else word  

if __name__ == "__main__":
    dictionary = load_dictionary("dictionary.txt")
    query = "befroe"
    corrected_query = " ".join(correct_word_ngram(word, dictionary) for word in query.split())
    print(f"Corrected Query: {corrected_query}")
