import pymorphy3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords




def preprocess(text: str) -> list[str]:
    punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...']
    stop_words = stopwords.words("russian")
    morph = pymorphy3.MorphAnalyzer()
    tokens = word_tokenize(text.lower(), language="russian")

    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return preprocessed_text


text_normalizator = preprocess