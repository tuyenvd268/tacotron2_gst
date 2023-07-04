import json

def load_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        _dict = json.load(f)
    return _dict

_char2index = load_dict(path="resources/char2index.json")
_index2char = load_dict(path="resources/index2char.json")
_word2phoneme = load_dict(path="resources/word2phoneme.json")

def word_to_phoneme(text):
    texts = text.split()
    texts = [_word2phoneme[text] if text in _word2phoneme else text for text in texts ]
    
    return " <spc> ".join(texts)
    

def text_to_ids(text):
    tokens = text.split()
    token_ids = [int(_char2index[token]) for token in tokens]
    
    return token_ids