from fastapi import Depends, Body
from fastapi.exceptions import ValidationException

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Annotated, Union

from uuid import uuid4

from .authorization import token_header


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionsBody(BaseModel):
    type: Optional[Literal["chat", "agent", "workflow"]] = Field(
        default=None,
        description="应用类型"
    )
    model: str = Field(description="要使用的模型的 ID", example="dify")
    conversation_id: Optional[str] = Field(default=None,
                                           description="会话 ID，需要基于之前的聊天记录继续对话，必须传之前消息的 conversation_id。")
    auto_generate_name: Optional[bool] = Field(default=True,
                                               description="自动生成标题，默认 true。 若设置为 false，则可通过调用会话重命名接口并设置 auto_generate 为 "
                                                           "true 实现异步生成标题。")
    inputs: Dict = Field(default={},
                         description="允许传入 App 定义的各变量值。 inputs 参数包含了多组键值对（Key/Value pairs），每组的键对应一个特定变量，每组的值则是该变量的具体值")
    user: Optional[str] = Field(
        default=str(uuid4()),
        description="用户标识，用于定义终端用户的身份，方便检索、统计。 由开发者定义规则，需保证用户标识在应用内唯一。如果不传默认随机生成一个 UUID"
    )
    detail: Optional[bool] = Field(default=False, description="是否返回详细信息（usage）")
    stream: Optional[bool] = Field(default=False, description="是否使用流式模式返回")
    messages: List[Message] = Field(description="消息列表", min_length=1)


def get_application_type(
        body: Annotated[ChatCompletionsBody, Body()],
        token: str = Depends(token_header)
):
    if body.type is not None:
        return body

    _type = token.split(":")[0]

    if _type is None:
        raise ValidationException("token 格式错误，需要在 token header 中或 body 中传入 type 参数")

    body.type = _type
    return body
