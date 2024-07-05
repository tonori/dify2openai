from fastapi import Depends
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials
from error.exception import UnauthorizedException
from typing import Optional

token_header = APIKeyHeader(
    name="Authorization",
    auto_error=False
)


def get_token(token: Optional[str] = Depends(token_header)):
    if token is None:
        raise UnauthorizedException()

    return token.replace("Bearer ", "")
