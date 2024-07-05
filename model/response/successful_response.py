from pydantic import BaseModel
from typing import List


class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str
    data: List[Model]
