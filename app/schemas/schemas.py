from pydantic import BaseModel

class McForSearchSchema(BaseModel):
    '''Схема для парсинга микрокатегорий из csv файла с ключевыми словами и остальной информацией о микрокатегории.'''
    mcId: int
    mcTitle: str
    keyPhrases: list[str]
    description: str