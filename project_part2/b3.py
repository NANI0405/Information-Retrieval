import json
import re


def load_dictionary(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def enhanced_soundex(word):
    word = word.strip().upper()

    if not word:
        return "0000"  

    soundex_dict = {
        "BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", 
        "MN": "5", "R": "6"
    }

    word = re.sub(r'[HW]', '', word) 

    if not word:
        return "0000"  

    first_letter = word[0]
    encoded = first_letter

    for char in word[1:]:
        for key, value in soundex_dict.items():
            if char in key:
                if not encoded.endswith(value):
                    encoded += value

    return encoded.ljust(4, '0')[:4]  


def correct_word_soundex(word, dictionary):
    if not word:
        return word  

    word_soundex = enhanced_soundex(word)
    best_match, best_score = None, 0

    for dict_word in dictionary:
        dict_soundex = enhanced_soundex(dict_word)
        score = sum(1 for x, y in zip(word_soundex, dict_soundex) if x == y)
        if score > best_score:
            best_match, best_score = dict_word, score

    return best_match if best_match else word  

if __name__ == "__main__":
    dictionary = load_dictionary("dictionary.txt")
    query = "befroe"
    corrected_query = " ".join(correct_word_soundex(word, dictionary) for word in query.split())
    print(f"Corrected Query: {corrected_query}")
