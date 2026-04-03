from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    # FastAPI server
    host: str = "0.0.0.0"
    port: int = 8080
    origins: list[str]

    # Dataset
    path_to_mc_search_dataset: str
    encoding_to_mc_search_dataset_csv: str

    # Google AI Studio
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash-lite"


settings = Settings()
