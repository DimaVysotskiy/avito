from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core.settings import settings
from .mc_search_dataset import McSearchDataset




@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mc_dataset = McSearchDataset(path=settings.path_to_mc_search_dataset, encoding=settings.encoding_to_mc_search_dataset_csv)
    yield
    

app = FastAPI(lifespan=lifespan)


@app.get("/microcategories")
async def get_microcategories():
    dataset = app.state.mc_dataset
    data = dataset.get_data()
    return data
