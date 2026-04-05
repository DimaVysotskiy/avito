import time
import logging

from fastapi import FastAPI, Body, Request, Response
from contextlib import asynccontextmanager

from .core.settings import settings
from .core.mc_reference import McReference
from .core.detector import McCandidateDetector
from .core.llm_usage import llm_usage
from .schemas import SplitPredictionRequest, SplitPredictionResponse

from google import genai

logger = logging.getLogger("uvicorn.error")
# Отключаем стандартный access-лог uvicorn, чтобы не дублировать
logging.getLogger("uvicorn.access").disabled = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Справочник и детектор ---
    try:
        mc_dataset = McReference(
            path=settings.path_to_mc_search_dataset,
            encoding=settings.encoding_to_mc_search_dataset_csv,
        )
        app.state.detector = McCandidateDetector(mc_dataset.get_data())
        logger.info("Lifespan  | Детектор инициализирован, микрокатегорий: %d", len(mc_dataset.get_data()))
    except Exception:
        logger.exception("Lifespan  | Не удалось инициализировать детектор")
        raise

    # --- Gemini клиент ---
    try:
        proxy = settings.proxy_url
        logger.info(
            "Lifespan  | Gemini: model=%s  proxy=%s",
            settings.gemini_model,
            proxy or "off",
        )

        client_kwargs = {"api_key": settings.gemini_api_key}
        if proxy:
            client_kwargs["http_options"] = {
                "async_client_args": {"proxy": proxy},
            }

        client = genai.Client(**client_kwargs)

        # Проверяем что клиент реально может достучаться до API
        models = await client.aio.models.list(config={"page_size": 1})
        logger.info("Lifespan  | Gemini клиент подключён ✓ (API доступен)")

        app.state.llm = client
    except Exception:
        logger.exception("Lifespan  | Не удалось подключиться к Gemini API")
        raise

    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)

    # Для /predict — читаем и логируем тело ответа
    if request.url.path == "/predict":
        body_chunks = [chunk async for chunk in response.body_iterator]
        body = b"".join(body_chunks)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            '%s:%s - "%s %s" %d  %.0fms  body=%s',
            request.client.host,
            request.client.port,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            body.decode(),
        )
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    # Остальные эндпоинты — обычный лог без тела
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        '%s:%s - "%s %s" %d  %.0fms',
        request.client.host,
        request.client.port,
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response

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
    # Первичная работа детектора над текстом
    detector: McCandidateDetector = app.state.detector

    logger.info(
        "Detector | itemId=%d  mcId=%d  text_len=%d",
        request.itemId, request.mcId, len(request.description),
    )

    detector_response = detector.detect(
        raw_text=request.description,
        source_mc_id=request.mcId,
    )

    # Нет кандидатов — ранний выход без LLM
    if not detector_response:
        logger.info("Detector | itemId=%d  candidates=0 → skip LLM", request.itemId)
        return SplitPredictionResponse(
            detectedMcIds=[],
            shouldSplit=False,
            drafts=[],
        )

    logger.info(
        "Detector | itemId=%d  candidates=%d  mc_ids=%s",
        request.itemId,
        len(detector_response.detected_mc),
        detector_response.detectedMcIds,
    )

    # LLM анализирует кандидатов и генерирует черновики
    client = app.state.llm
    draft_response = await llm_usage(
        client=client,
        request=request,
        candidates=detector_response.detected_mc,
    )

    return SplitPredictionResponse(
        detectedMcIds=draft_response.detectedMcIds,
        shouldSplit=draft_response.shouldSplit,
        drafts=draft_response.drafts,
    )