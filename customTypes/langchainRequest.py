from pydantic import BaseModel, Field

class langchainRequest(BaseModel):
    query: str = Field(title="Query", description="Input text")