from fastapi import status
from fastapi.exceptions import HTTPException


class UnauthorizedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class ValidationException(HTTPException):
    message: str

    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

        self.message = message
