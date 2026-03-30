from pydantic_settings import BaseSettings, SettingsConfigDict




class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # В .env можно писать POSTGRES__USER=admin , приставки берутся по названию полей в Settings
    )
    origins: list[str]

    # FastAPI server 
    host: str = "0.0.0.0"
    port: int = 8080
    origins: list[str]


    path_to_mc_search_dataset: str
    encoding_to_mc_search_dataset_csv: str


    

settings = Settings()