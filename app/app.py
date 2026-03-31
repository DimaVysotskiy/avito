from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core.settings import settings
from .mc_search_dataset import McSearchDataset
from .schemas import SplitPredictionRequest
from .core.text_normalizator import text_normalizator




@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mc_dataset = McSearchDataset(path=settings.path_to_mc_search_dataset, encoding=settings.encoding_to_mc_search_dataset_csv)
    yield
    

app = FastAPI(lifespan=lifespan)



@app.post("/main")

async def main(request: SplitPredictionRequest):
    norm_text = text_normalizator(request.description)
    return norm_text



@app.get("/microcategories")
async def get_microcategories():
    dataset = app.state.mc_dataset
    data = dataset.get_data()
    return data
