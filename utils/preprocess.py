import string
from nltk.stem import PorterStemmer
import json
import re
# from functools import lru_cache  MAY USE LATER TO CAHE GET STOP WORDS FUNCTION FOR NOW JUST DO IT GLOBALLY

def load_stopwords() -> list[str]:
    with open("./data/stopwords.txt", "r") as f:
        return f.read().splitlines()

def preprocess(s : str) -> list[str]:
    # print(s,end=" ")
    s = s.lower()
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator)
    tokens = s.split()
    valid_tokens = [] # may be set so only unique token
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_token = remove_stopWords(valid_tokens)
    stemmer = PorterStemmer()
    stemmed_token = []
    for ft in filtered_token:
        stemmed_token.append(stemmer.stem(ft))
    return stemmed_token

def remove_stopWords(tokens) ->list :
    stop_words = load_stopwords()
    filtered_words = []
    for word in tokens:
        if word not in stop_words:
            filtered_words.append(word)
    return filtered_words



def safe_json_loads(text):
    ### might want to add pydantic here later (MAY BE)
    """
    Safely parse a model response that might contain JSON in Markdown or plain text.
    Returns parsed JSON list or raises ValueError.
    """
    # Remove markdown fences like ```json ... ``` because some reason model just likes to add it
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nRaw text:\n{text}")
