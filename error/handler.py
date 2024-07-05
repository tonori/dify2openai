from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from error.exception import UnauthorizedException, ValidationException
from model.response.error_response import UnauthorizedResponse, ValidationErrorResponse


def unauthorized_error_handler(_, __):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=UnauthorizedResponse().model_dump()
    )


def validation_error_handler(_, exc: ValidationException):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ValidationErrorResponse(
            message=exc.message
        ).model_dump()
    )


def error_handler_register(app: FastAPI):
    app.add_exception_handler(UnauthorizedException, unauthorized_error_handler)
    app.add_exception_handler(ValidationException, validation_error_handler)
