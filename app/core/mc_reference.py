import csv
from typing import List, Any
from ..schemas.schemas import McForSearchSchema, McForSearchSchemaNorm
from .text_normalizator import text_normalizator




class McReference:
    '''Справочник с информацией по микрокатегориям'''
    def __init__(self, path: str, encoding: str = 'utf-8'):
        self.path = path
        self.encoding = encoding
        self.data: List[McForSearchSchema] = []
        self.norm_data: List[dict[str, Any]] = []
        self._load_csv()
        self._normalize_key_phrases()


    def _load_csv(self):
        """Загрузка CSV с учетом кавычек для полей с запятыми"""
        with open(file=self.path, mode='r', encoding=self.encoding, newline='') as file:
            reader = csv.DictReader(file, quotechar='"', delimiter=',')
            for row in reader:
                key_phrases = [kp.strip() for kp in row['keyPhrases'].split(';')] if row.get('keyPhrases') else []
                self.data.append(McForSearchSchema(
                    mcId=row['mcId'],
                    mcTitle=row['mcTitle'],
                    keyPhrases=key_phrases,
                    description=row.get('description', '')
                ))


    def _normalize_key_phrases(self) -> None:
        self.norm_data = []
        for item in self.data:
            self.norm_data.append(
            McForSearchSchemaNorm(
                    mcId=item.mcId,
                    mcTitle=item.mcTitle,
                    keyPhrases=[text_normalizator(phrase) for phrase in item.keyPhrases],
                    description=item.description
                ))




    def get_data(self) -> List[McForSearchSchema]:
        return self.data
    

    def get_norm_data(self) -> List[McForSearchSchemaNorm]:
        return self.norm_data