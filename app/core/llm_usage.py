import json

from google.genai import Client
from ..schemas import SplitPredictionRequest, CandidateMc, DraftResponse, InfoToLLM
from .settings import settings




SYSTEM_PROMPT = """Ты — эксперт по классификации объявлений на сайте с объявлениями, категория «Ремонт и отделка».

Тебе передают json формата:
{
  "main_mc_id": <уникальный id основной микрокатегории>,
  "main_mc_title": "<название основной микрокатегории>",
  "description": "<описание объявления>",
  "candidates_ids": [<уникальный id микрокатегории кандидата на разделение>, ...],
  "candidates": [
    {
      "mc_id": <уникальный id микрокатегории кандидата на разделение>,
      "mc_title": "<название микрокатегории кандидата на разделение>",
      "matched_phrases": ["<ключевая фраза по которой детектор нашел совпадение>", ...]
    }
  ]
}

Твоя задача — для каждого кандидата определить, оказывает ли исполнитель эту услугу комлексно с основной услугой или отдельно как самостоятельную услугу, или она упомянута лишь как часть комплексной работы и самостоятельно не выполняется.

Критерии для самостоятельной услуги (добавляем в drafts):
- явные сигналы: «отдельно», «также выполняем», «самостоятельная услуга», перечисление услуг как независимых
- по контексту и смыслу объявления понятно что услуга может выполняться отдельно от основной

Критерии против (не добавляем в drafts):
- «включая», «в том числе», «под ключ включает», «в составе работ» — услуга является частью комплекса
- строгий контекст комплексных работ

Для каждой микрокатегории, которую стоит выделить, напиши черновик объявления на основе исходного текста и соблюдая стиль исходного объявления. Черновик должен:
- быть написан от первого лица (как исполнитель)
- сохранять стиль исходного объявления
- содержать только информацию, релевантную данной микрокатегории
- быть лаконичным и конкретным (3–7 предложений)

Отвечай СТРОГО в формате JSON (без markdown, без пояснений вне JSON):
{
  "detectedMcIds": [<все mcId кандидатов, найденных в тексте + mcId основной микрокатегории>],
  "shouldSplit": <true если хотя бы один кандидат выделяется как самостоятельная услуга>,
  "targetSplitMcIds": [<mcId кандидатов, которые выделяются как самостоятельная услуга>],
  "drafts": [
    {
      "mcId": <int>,
      "mcTitle": "<string>",
      "text": "<черновик объявления>"
    }
  ]
}
Если ни один кандидат не является самостоятельной услугой — верни поля targetSplitMcIds=[], drafts=[],shouldSplit=false и остальные поля в соответствии с форматом.

Отвечай только в формате JSON, без markdown, без пояснений вне JSON!!!
"""




def build_user_prompt(
    request: SplitPredictionRequest,
    candidates: list[CandidateMc],
) -> str:
    """Формирует JSON-строку для пользовательского промпта из запроса и кандидатов."""
    info = InfoToLLM(
        main_mc_id=request.mcId,
        main_mc_title=request.mcTitle,
        description=request.description,
        candidates_ids=[c.mc_id for c in candidates],
        candidates=candidates,
    )
    return info.model_dump_json(indent=2)




async def llm_usage(
    genai_client: Client,
    request: SplitPredictionRequest,
    candidates: list[CandidateMc],
) -> DraftResponse:
    """
    Отправляет запрос к Gemini и возвращает структурированный ответ.

    Клиент `genai_client` создаётся один раз в lifespan и переиспользуется
    из `app.state.llm` без повторной инициализации.
    """
    user_prompt = build_user_prompt(request, candidates)

    response = await genai_client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=user_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
        },
    )

    raw_text = response.text.strip()
    parsed = json.loads(raw_text)

    return DraftResponse(**parsed)