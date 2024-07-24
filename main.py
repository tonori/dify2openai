import uvicorn
import json
import re
from env import env
from fastapi import FastAPI, APIRouter, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Dict, Optional, List, Literal, Annotated
from typing_extensions import TypedDict
from dependencies.authorization import get_token
from dependencies.get_application_type import get_application_type
from error.exception import ValidationException
from error.handler import error_handler_register
from model.response.successful_response import ModelsResponse, Model
from datetime import datetime
from uuid import uuid4
from api.chat_messages import ChatMessageBody, request_chat_messages, request_stream_chat_message

openai_routes = APIRouter(
    prefix="/v1",
    tags=["OpenAI"]
)

app = FastAPI(
    title="Dify2OpenAI",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

error_handler_register(app)


class AreYouOk(BaseModel):
    message: str = Field(
        default="I'm OK!",
        description="How are you?"
    )


@app.get("/are-you-ok", response_model=AreYouOk)
async def are_you_ok():
    return AreYouOk().model_dump()


@openai_routes.get(
    "/models",
    response_model=ModelsResponse
)
async def get_models(
        token: str = Depends(get_token)
):
    return ModelsResponse(
        object="list",
        data=[
            Model(
                id="dify",
                object="model",
                created=int(datetime.now().timestamp()),
                owned_by="dify"
            )
        ]
    ).model_dump()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = Field(
        description="提示词中的 Token 数量",
    )
    completion_tokens: int = Field(
        description="生成的文本中的 Token 数量",
    )
    total_tokens: int = Field(
        description="提示词和生成的文本的总 Token 数量",
    )


class ChatCompletionsChoice(BaseModel):
    finish_reason: Literal["stop", "length", "content_filter"] = Field(
        description="停止原因"
    )
    index: int = Field(description="索引")
    message: Message = Field(description="消息内容")


class ChatCompletionsResponse(BaseModel):
    id: str = Field(description="消息唯一 ID", example="443b04c3-8c1f-4cbb-9bd2-8890426f410a")
    object: str = Field(description="对象类型，始终是 chat.completion", example="chat.completion")
    created: int = Field(description="消息完成创建时的 Unix 时间戳（以秒为单位）", example=1705395332)
    model: str = Field(description="模型 ID", example="dify")
    choices: List[ChatCompletionsChoice] = Field(description="消息列表")
    usage: ChatCompletionUsage = Field(description="用量")
    conversation_id: str = Field(description="会话 ID")
    retriever_resources: Optional[list] = Field(description="引用和归属分段列表")


class ChatCompletionsStreamChoiceDelta(BaseModel):
    role: Optional[Literal["assistant"]] = Field(default=None)
    content: Optional[str] = Field(default=None)


class ChatCompletionsStreamChoice(BaseModel):
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = Field(
        default=None,
        description="停止原因"
    )
    index: int = Field(description="索引")
    delta: ChatCompletionsStreamChoiceDelta = Field(description="文本增量内容", default={})


class ChatCompletionsStreamChunk(BaseModel):
    id: str = Field(description="消息唯一 ID", example="443b04c3-8c1f-4cbb-9bd2-8890426f410a")
    conversation_id: str = Field(description="会话 ID")
    task_id: str = Field(description="任务 ID，用于请求跟踪和请求停止响应接口")
    object: Literal["chat.completion.chunk", "chat.completion.usage"] = Field(description="对象类型",
                                                                              example="chat.completion.chunk")
    created: int = Field(description="消息完成创建时的 Unix 时间戳（以秒为单位）", example=1705395332)
    model: str = Field(description="模型 ID", example="dify")
    choices: List[ChatCompletionsStreamChoice] = Field(description="消息列表")
    usage: Optional[ChatCompletionUsage] = Field(default=None, description="用量")


class DifyMessageEventMetadata(TypedDict):
    usage: Optional[ChatCompletionUsage]


class DifyMessageEventChunk(TypedDict):
    event: Literal["message", "message_end"]
    conversation_id: str
    message_id: str
    created_at: int
    task_id: str
    id: str
    answer: str
    metadata: Optional[DifyMessageEventMetadata]


async def stream_completions_handler(
        request_data: ChatMessageBody,
        token: str,
        model: str
):
    async with request_stream_chat_message(
            body=request_data,
            token=token
    ) as _response:
        event_regex = re.compile(r"data: (.*)\n\n")

        index = 0

        async for chunk in _response.aiter_bytes():
            _match = event_regex.match(chunk.decode("utf-8"))

            if _match is None:
                continue

            event_data = json.loads(_match.group(1))

            _event_data: DifyMessageEventChunk = event_data

            is_last_chunk = _event_data.get("event") == "message_end"

            data = ChatCompletionsStreamChunk(
                id=_event_data.get("message_id"),
                conversation_id=_event_data.get("conversation_id"),
                task_id=_event_data.get("task_id"),
                object="chat.completion.chunk" if is_last_chunk is False else "chat.completion.usage",
                created=_event_data.get("created_at"),
                model=model,
                choices=[
                    ChatCompletionsStreamChoice(
                        index=index,
                        delta=ChatCompletionsStreamChoiceDelta(
                            role="assistant" if index == 0 else None,
                            content=_event_data.get("answer")
                        ) if is_last_chunk is False else {},
                        finish_reason="stop" if is_last_chunk else None
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=_event_data.get("metadata", {}).get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=_event_data.get("metadata", {}).get("usage", {}).get("completion_tokens",
                                                                                           0),
                    total_tokens=_event_data.get("metadata", {}).get("usage", {}).get("total_tokens", 0)
                ) if is_last_chunk else None
            ).model_dump_json()

            yield f'data: {data}\n\n'.encode()

            index += 1


@openai_routes.post(
    "/chat/completions",
    response_model=ChatCompletionsResponse
)
async def chat_completions(
        body=Depends(get_application_type),
        token: str = Depends(get_token),
):
    request_data = ChatMessageBody(
        query=list(filter(lambda message: message.role == "user", body.messages))[-1].content,
        inputs=body.inputs,
        response_mode="streaming" if body.stream else "blocking",
        user=body.user,
        conversation_id=body.conversation_id,
        attributes=body.auto_generate_name
    )

    if body.stream:
        return StreamingResponse(
            content=stream_completions_handler(
                request_data=request_data,
                token=token,
                model=body.model
            ),
            media_type="text/event-stream"
        )

    model_response = await request_chat_messages(
        body=request_data,
        token=token
    )

    response = ChatCompletionsResponse(
        conversation_id=model_response.get("conversation_id"),
        id=model_response.get("message_id"),
        object="chat.completion",
        created=model_response.get("created_at"),
        model=body.model,
        choices=[
            ChatCompletionsChoice(
                finish_reason="stop",
                index=0,
                message=Message(
                    role="assistant",
                    content=model_response.get("answer")
                )
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=model_response["metadata"]["usage"]["prompt_tokens"],
            completion_tokens=model_response["metadata"]["usage"]["completion_tokens"],
            total_tokens=model_response["metadata"]["usage"]["total_tokens"]
        ),
        retriever_resources=model_response["metadata"].get("retriever_resources")
    )

    return response.model_dump()


app.include_router(openai_routes)

if __name__ == "__main__":
    uvicorn.run("main:app", host=env.str("SERVER_HOST", default="127.0.0.1"),
                port=env.int("SERVER_PORT", default=8000), reload=True)
