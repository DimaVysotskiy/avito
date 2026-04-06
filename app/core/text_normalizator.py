import re
import logging
import unicodedata
import pymorphy3
import spacy
from functools import lru_cache

logger = logging.getLogger("uvicorn.error")

_morph = pymorphy3.MorphAnalyzer()

# Используем пустую русскую модель SpaCy для очень быстрой C-токенизации
# Это работает без загрузки тяжелых моделей и умеет определять стоп-слова и пунктуацию
nlp = spacy.blank("ru")


@lru_cache(maxsize=10000)
def _get_lemma(token: str) -> str:
    """Кэшируемая версия лемматизации для ускорения обработки повторяющихся слов."""
    return _morph.parse(token)[0].normal_form


def _clean(text: str) -> str:
    """Заменяем разделители и мусор на пробелы, нормализуем unicode."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[/\\|;+]', ' ', text)   # слеши и разделители → пробел
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess(text: str) -> list[str]:
    cleaned = _clean(text)
    logger.debug(f"Normalizer | Очищенный текст: '{cleaned}'")

    doc = nlp(cleaned.lower())
    
    # Только для логов собираем строковые представления токенов
    tokens_text = [t.text for t in doc]
    logger.debug(f"Normalizer | Токены ({len(tokens_text)}): {tokens_text}")

    result = []
    for token in doc:
        # Быстрая проверка на пунктуацию, пробелы и стоп-слова на уровне C-кода (Cython)
        if token.is_punct or token.is_space or token.is_stop:
            continue
            
        lemma = _get_lemma(token.text)
        result.append(lemma)

    logger.debug(f"Normalizer | Леммы ({len(result)}): {result}")
    return result


text_normalizator = preprocess