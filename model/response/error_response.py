from pydantic import BaseModel
from typing import Optional


class ErrorResponse(BaseModel):
    message: str
    type: str
    param: Optional[str]
    code: Optional[str]


class UnauthorizedResponse(ErrorResponse):
    def __init__(self):
        super().__init__(
            message="You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY). You can obtain an API key from https://docs.dify.ai/guides/application-publishing/developing-with-apis#how-to-use.",
            type="invalid_request_error",
            param=None,
            code=None
        )


class ValidationErrorResponse(ErrorResponse):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            type="invalid_request_error",
            param=None,
            code=None
        )
