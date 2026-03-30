import csv
from typing import List
from ..schemas import McForSearchSchema 

class McSearchDataset:
    def __init__(self, path: str, encoding: str = 'utf-8'):
        self.path = path
        self.encoding = encoding
        self.data: List[McForSearchSchema] = []
        self._load_csv()

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

    def get_data(self) -> List[McForSearchSchema]:
        return self.data