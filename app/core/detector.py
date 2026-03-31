from dataclasses import dataclass, field
from typing import List

from ..schemas import McForSearchSchema
from .text_normalizator import text_normalizator


@dataclass
class CandidateMc:
    mc_id: int
    mc_title: str
    matched_phrases: List[str] = field(default_factory=list)


class McCandidateDetector:
    """
    Инициализируется один раз при старте приложения.
    Кэширует лемматизированные ключевые фразы всех микрокатегорий.
    """
    def __init__(self, mc_data: List[McForSearchSchema]):
        self._mc_titles: dict[int, str] = {}
        self._mc_phrases: dict[int, list[tuple[list[str], str]]] = {} # { mc_id : ([леммы фразы], оригинал фразы) }

        for mc in mc_data:
            self._mc_titles[mc.mcId] = mc.mcTitle
            self._mc_phrases[mc.mcId] = [
                (text_normalizator(phrase), phrase)
                for phrase in mc.keyPhrases
                if phrase.strip()
            ]


    def detect(self, raw_text: str, source_mc_id: int) -> List[CandidateMc]:
        """
        Ищем все МК (кроме source_mc_id) в тексте объявления.
        Возвращаем кандидатов, отсортированных по количеству совпавших фраз.
        """
        lemmas = text_normalizator(raw_text)
        n = len(lemmas)

        candidates: dict[int, CandidateMc] = {}

        for mc_id, phrases in self._mc_phrases.items():
            if mc_id == source_mc_id:
                continue

            for phrase_lemmas, phrase_orig in phrases:
                k = len(phrase_lemmas)
                if k == 0 or k > n:
                    continue

                for i in range(n - k + 1):
                    if lemmas[i:i + k] == phrase_lemmas:
                        if mc_id not in candidates:
                            candidates[mc_id] = CandidateMc(
                                mc_id=mc_id,
                                mc_title=self._mc_titles[mc_id],
                            )
                        candidates[mc_id].matched_phrases.append(phrase_orig)
                        break  # одно вхождение фразы достаточно

        return sorted(candidates.values(), key=lambda c: len(c.matched_phrases), reverse=True)