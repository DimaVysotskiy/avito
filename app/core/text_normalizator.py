import re
import logging
import unicodedata
import pymorphy3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger("uvicorn.error")


_morph = pymorphy3.MorphAnalyzer()
_stop_words = set(stopwords.words("russian"))
_punct = {'!', ',', '(', ')', ':', '-', '?', '.', '..', '...'}




def _clean(text: str) -> str:
    """Заменяем разделители и мусор на пробелы, нормализуем unicode."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[/\\|;+]', ' ', text)   # слеши и разделители → пробел
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess(text: str) -> list[str]:
    cleaned = _clean(text)
    logger.debug("Normalizer | Очищенный текст: '%s'", cleaned)

    tokens = word_tokenize(cleaned.lower(), language="russian")
    logger.debug("Normalizer | Токены (%d): %s", len(tokens), tokens)

    result = []
    for token in tokens:
        if token in _punct:
            continue
        lemma = _morph.parse(token)[0].normal_form
        if lemma not in _stop_words:
            result.append(lemma)

    logger.debug("Normalizer | Леммы (%d): %s", len(result), result)
    return result




text_normalizator = preprocess