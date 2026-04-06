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
            self._mc_phrases[mc.mcId] = [
                (text_normalizator(phrase), phrase)
                for phrase in mc.keyPhrases
                if phrase.strip()
            ]


    def detect(self, raw_text: str, source_mc_id: int) -> DetectorResponse | None:

        logger.info(f"Detector | Нормализация текста ({len(raw_text)} символов)...")
        lemmas = text_normalizator(raw_text)
        logger.info(f"Detector | Леммы ({len(lemmas)}): {lemmas}")

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
                                matched_phrases=[]
                            )
                        candidates[mc_id].matched_phrases.append(phrase_orig)
                        logger.debug(
                            f"Detector | Совпадение: mc_id={mc_id} ({self._mc_titles[mc_id]})  фраза='{phrase_orig}'  леммы={phrase_lemmas}  позиция={i}",
                        )
                        break
        
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