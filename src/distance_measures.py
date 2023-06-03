from collections import Counter


def out_of_place(doc_profile: Counter, cat_profile: Counter) -> int:
    score = 0

    doc_ranks = {word: rank for rank, word in enumerate(doc_profile)}
    cat_ranks = {word: rank for rank, word in enumerate(cat_profile)}

    for word in doc_ranks:
        score += abs(doc_ranks[word] - cat_ranks.get(word, len(cat_ranks)))
    return score
