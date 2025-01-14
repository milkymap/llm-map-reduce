from pydantic_settings import BaseSettings
from pydantic import Field 

class Credentials(BaseSettings):
    openai_api_key:str=Field(validation_alias='OPENAI_API_KEY', required=True)