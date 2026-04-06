import logging

from fastapi import FastAPI, Body
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager

from .core.settings import settings
from .core.mc_reference import McReference
from .core.detector import McCandidateDetector
from .core.llm_usage import llm_usage
from .schemas import SplitPredictionRequest, SplitPredictionResponse

from google import genai




logger = logging.getLogger("uvicorn.error")




@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Справочник и детектор ---
    try:
        mc_dataset = McReference(
            path=settings.path_to_mc_search_dataset,
            encoding=settings.encoding_to_mc_search_dataset_csv,
        )
        app.state.detector = McCandidateDetector(mc_dataset.get_data())
        logger.info(f"Lifespan  | Детектор инициализирован, микрокатегорий: {len(mc_dataset.get_data())}")
    except Exception:
        logger.exception("Lifespan  | Не удалось инициализировать детектор")
        raise


    # --- Gemini клиент ---
    try:
        proxy = settings.proxy_url

        logger.info(
            f"Lifespan  | Gemini: model={settings.gemini_model}  proxy={proxy or 'off'}",
        )


        client_kwargs = {"api_key": settings.gemini_api_key}

        if proxy:
            client_kwargs["http_options"] = {
                "async_client_args": {"proxy": proxy},
            }

        genai_client = genai.Client(**client_kwargs)

        # Проверяем что клиент реально может достучаться до API
        models = await genai_client.aio.models.list(config={"page_size": 1})
        logger.info("Lifespan  | Gemini клиент подключён ✓ (API доступен)")

        app.state.llm = genai_client
    except Exception:
        logger.exception("Lifespan  | Не удалось подключиться к Gemini API")
        raise

    yield




app = FastAPI(lifespan=lifespan)




PREDICT_EXAMPLE = {
    "default": {
        "summary": "Ремонт под ключ с отдельными услугами",
        "value": {
            "itemId": 8,
            "mcId": 101,
            "mcTitle": "Ремонт квартир и домов под ключ",
            "description": "Полное обновление квартиры в доме. Отдельно гидроизоляция под плитку делаем отдельно вызов сантехника можем отдельно установка выключателей отдельно покраска дверей. работаем с 2016 года. без скрытых доплат. оперативный выезд.",
        },
    }
}


@app.post("/predict", response_model=SplitPredictionResponse)
async def predict(
    request: SplitPredictionRequest = Body(openapi_examples=PREDICT_EXAMPLE),
):
    # Первичная работа детектора над текстом (передана в потоки, чтобы не блокировать event loop CPU-вычислениями)
    detector: McCandidateDetector = app.state.detector

    logger.info(
        f"Detector | itemId={request.itemId}  mcId={request.mcId}  text_len={len(request.description)}",
    )


    detector_response = await run_in_threadpool(
        detector.detect,
        raw_text=request.description,
        source_mc_id=request.mcId,
    )

    # Нет кандидатов — ранний выход без LLM
    if not detector_response:
        logger.info(f"Detector | itemId={request.itemId}  candidates=0 → skip LLM")
        return SplitPredictionResponse(
            detectedMcIds=[],
            shouldSplit=False,
            drafts=[],
        )

    logger.info(
        f"Detector | itemId={request.itemId}  candidates={len(detector_response.detected_mc)}  mc_ids={detector_response.detectedMcIds}",
    )

    # LLM анализирует кандидатов и генерирует черновики
    genai_client = app.state.llm
    draft_response = await llm_usage(
        genai_client=genai_client,
        request=request,
        candidates=detector_response.detected_mc,
    )

    return SplitPredictionResponse(
        detectedMcIds=draft_response.detectedMcIds,
        shouldSplit=draft_response.shouldSplit,
        drafts=draft_response.drafts,
    )