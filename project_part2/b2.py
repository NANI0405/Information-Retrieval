import json
import numpy as np


def load_dictionary(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def edit_distance(word1, word2):
    dp = np.zeros((len(word1) + 1, len(word2) + 1))

    for i in range(len(word1) + 1):
        for j in range(len(word2) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len(word1)][len(word2)]


def correct_word_edit_distance(word, dictionary):
    return min(dictionary, key=lambda w: edit_distance(word, w))

if __name__ == "__main__":
    dictionary = load_dictionary("dictionary.txt")
    query = "befroe"
    corrected_query = " ".join(correct_word_edit_distance(word, dictionary) for word in query.split())
    print(f"Corrected Query: {corrected_query}")
