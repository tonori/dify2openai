from .client import httpx_client
from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal, List
from typing_extensions import TypedDict


class ChatMessageFile(BaseModel):
    type: Literal["image"] = Field(
        description="支持类型：图片 image（目前仅支持图片格式）"
    )
    transfer_method: Literal["remote_url", "local_file"] = Field(
        description="传递方式。remote_url：图片地址；local_file：上传文件"
    )
    url: Optional[str] = Field(description="图片地址。（仅当传递方式为 remote_url 时）。")
    upload_file_id: Optional[str] = Field(
        description="上传文件 ID。（仅当传递方式为 local_file 时）。"
    )


class ChatMessageBody(BaseModel):
    query: str = Field(
        description="用户输入/提问内容。"
    )
    inputs: Optional[Dict] = Field(
        default={},
        description="允许传入 App 定义的各变量值。 inputs 参数包含了多组键值对（Key/Value pairs），每组的键对应一个特定变量，每组的值则是该变量的具体值。"
    )
    user: str = Field(
        description="用户标识，用于定义终端用户的身份，方便检索、统计。 由开发者定义规则，需保证用户标识在应用内唯一。"
    )
    response_mode: Optional[Literal["streaming", "blocking"]] = Field(
        default="blocking",
        description="响应模式。streaming：流式模式；blocking: 阻塞模式",
    )
    conversation_id: Optional[str] = Field(
        description="会话 ID，需要基于之前的聊天记录继续对话，必须传之前消息的 conversation_id。"
    )
    auto_generate_name: Optional[bool] = Field(
        default=True,
        description="自动生成标题，默认 true。 若设置为 false，则可通过调用会话重命名接口并设置 auto_generate 为 true 实现异步生成标题。"
    )
    files: Optional[List[ChatMessageFile]] = Field(
        default=None,
        description="上传的文件"
    )


class ChatCompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionMetadata(TypedDict):
    usage: ChatCompletionUsage
    retriever_resources: Optional[List[dict]]


class ChatCompletionResponse(TypedDict):
    message_id: str
    conversation_id: str
    mode: Literal["chat"]
    answer: str
    metadata: ChatCompletionMetadata
    created_at: int


async def request_chat_messages(body: ChatMessageBody, token: str) -> ChatCompletionResponse:
    response = await httpx_client.post(
        url="/chat-messages",
        headers={"Authorization": f"Bearer {token}"},
        json=body.model_dump()
    )
    return response.json()


def request_stream_chat_message(body: ChatMessageBody, token: str) -> httpx_client.stream:
    return httpx_client.stream(
        method="POST",
        url="/chat-messages",
        headers={"Authorization": f"Bearer {token}"},
        json=body.model_dump()
    )
