import logging
from typing import List
from ..schemas import McForSearchSchema, CandidateMc, DetectorResponse
from .text_normalizator import text_normalizator




logger = logging.getLogger("uvicorn.error")




class McCandidateDetector:
    def __init__(self, mc_data: List[McForSearchSchema]):
        self._mc_titles: dict[int, str] = {}
        self._mc_phrases: dict[int, list[tuple[list[str], str]]] = {}

        for mc in mc_data:
            self._mc_titles[mc.mcId] = mc.mcTitle
            phrases_list = []
            for phrase in mc.keyPhrases:
                if not phrase.strip():
                    continue
                lemmas = text_normalizator(phrase)
                if lemmas:
                    # Добавляем пробелы по краям для точного поиска границ слов
                    phrases_list.append((f" {' '.join(lemmas)} ", phrase))
            self._mc_phrases[mc.mcId] = phrases_list


    def detect(self, raw_text: str, source_mc_id: int) -> DetectorResponse | None:

        logger.info(f"Detector | Нормализация текста ({len(raw_text)} символов)...")
        lemmas = text_normalizator(raw_text)
        logger.info(f"Detector | Леммы ({len(lemmas)}): {lemmas}")

        if not lemmas:
            logger.info("Detector | Текст не содержит значимых слов")
            return None

        # Оборачиваем в пробелы для точного совпадения целых слов
        text_str = f" {' '.join(lemmas)} "
        candidates: dict[int, CandidateMc] = {}

        for mc_id, phrases in self._mc_phrases.items():
            if mc_id == source_mc_id:
                continue

            for phrase_lemmas_str, phrase_orig in phrases:
                # Быстрый поиск подстроки на уровне C вместо медленных срезов списков
                if phrase_lemmas_str in text_str:
                    if mc_id not in candidates:
                        candidates[mc_id] = CandidateMc(
                            mc_id=mc_id,
                            mc_title=self._mc_titles[mc_id],
                            matched_phrases=[]
                        )
                    candidates[mc_id].matched_phrases.append(phrase_orig)
                    logger.debug(
                        f"Detector | Совпадение: mc_id={mc_id} ({self._mc_titles[mc_id]})  фраза='{phrase_orig}'",
                    )
        
        if not candidates:
            logger.info("Detector | Кандидатов не найдено")
            return None

        sorted_candidates = sorted(candidates.values(), key=lambda c: len(c.matched_phrases), reverse=True)

        for c in sorted_candidates:
            logger.info(
                f"Detector | Кандидат mc_id={c.mc_id} ({c.mc_title})  совпадений={len(c.matched_phrases)}  фразы={c.matched_phrases}",
            )

        return DetectorResponse(
            detectedMcIds=[c.mc_id for c in sorted_candidates],
            detected_mc=sorted_candidates
        )