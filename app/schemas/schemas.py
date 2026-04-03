from pydantic import BaseModel

class McForSearchSchema(BaseModel):
    mcId: int
    mcTitle: str
    keyPhrases: list[str]
    description: str

class McForSearchSchemaNorm(BaseModel):
    mcId: int
    mcTitle: str
    keyPhrases: list[list[str]]
    description: str

class SplitPredictionRequest(BaseModel):
    itemId: int
    mcId: int
    mcTitle: str
    description: str

class CandidateMc(BaseModel):
    mc_id: int
    mc_title: str
    matched_phrases: list[str] = []

class DraftSchema(BaseModel):
    mcId: int
    mcTitle: str
    text: str

class SplitPredictionResponse(BaseModel):
    detectedMcIds: list[int]
    shouldSplit: bool
    drafts: list[DraftSchema]

class InfoToLLM(BaseModel):
    '''JSON со всей информацией о объявлении и кандидатах на разделение для LLM'''
    main_mc_id: int
    main_mc_title: str
    description: str
    candidates_ids: list[int]
    candidates: list[CandidateMc]

class DraftResponse(BaseModel):
    '''JSON ответ на запрос по выявлению микрокатегорий и написанию к ним черновиков'''
    detectedMcIds: list[int]
    targetSplitMcIds: list[int]
    shouldSplit: bool
    drafts: list[DraftSchema]

class DetectorResponse(BaseModel):
    '''JSON ответ на запрос по выявлению микрокатегорий от детоктора Detector ф-ция detect.
    Детектор только ищет по лемам, а LLM следующим шагом уже решает делить объявление на микро категории или нет.'''
    detectedMcIds: list[int]
    detected_mc: list[CandidateMc]