# Dify2OpenAI

将 Dify 应用 API 转化为 OpenAI 风格的 API

## 实现功能

### 聊天助手（Chat Bot）

- [x] 流式响应和非流式响应
- [ ] Dify API 异常响应
- [ ] 流式响应中的 message_replace（消息替换）& message_file（文件事件） & ping 事件
- [ ] 上传图像
- [ ] 工作流编排模式下的 Workflow 和 Node 相关的事件

### Agent

还没做- -

## Workflow

还没做- -

## 接口文档

[Developing with APIs - Dify](https://docs.dify.ai/guides/application-publishing/developing-with-apis)

[Swagger Doc](https://tonori.github.io/dify2openai/)

## 安装

### Conda

```shell
conda env create -f conda_env.yaml 
```

### pip

```shell
pip install -r requirements.txt
```

## 运行

```shell
python main.py
```

## 接口说明
### 认证
在 HTTP Authorization Header 中传递应用的 API Key 即可。

### 应用类型
| 标识（identifier） | 对应类型  |
|----------------|-------|
| chat           | 聊天助手  |
| agent          | Agent |
| workflow       | 工作流   |

~~为了适配 Dify 的应用类型响应差别，可以在 Authorization Header 中以 `{identifier}:{key}` 的格式指定应用类型。例如：`chat:app-xxxxxxxxxxxx` / `agent:app-xxxxxxxxxxxx` / `workflow:app-xxxxxxxxxxxx`~~ **(这还没做)**

~~也可以在 completions 接口的 Request JSON 中指定 `type` 参数，如果在 Authorization Header 和 Request JSON 中同时指定了应用类型，以 Request JSON 中的值为准。~~ **(这也还没做)**

```json
{
  "type": "chat",
  "model": "dify",
  "messages": [
    {
      "role": "string",
      "content": "string"
    }
  ]
}
```

### 上下文保持

Dify API 使用 `conversation_id` 进行上下文保持，conversation_id 在 ChatCompletionsResponse 和 ChatCompletionsStreamChunk 中均有返回。可以通过 Request JSON 的 `conversation_id` 参数进行指定。

如果在 Request JSON Messages 中传递了多条 Message，只会使用最后一条 **Role 为 user** 的 message 请求 Dify 接口。

## Response
### ChatCompletionsResponse
```json
{
  "id": "443b04c3-8c1f-4cbb-9bd2-8890426f410a",
  "object": "chat.completion",
  "created": 1705395332,
  "model": "dify",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "string",
        "content": "string"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "conversation_id": "string",
  "retriever_resources": [
    "string"
  ]
}
```

### ChatCompletionsStreamChunk
```json
{
    "id": "de80f5a9-6084-4324-9f78-92b51a91195b",
    "conversation_id": "0509851c-c6f3-4ebb-8319-cc045782c9fd",
    "task_id": "7fbb4a3f-079c-454f-a4cb-bca65adb60ca",
    "object": "chat.completion.chunk",
    "created": 1720233867,
    "model": "dify",
    "choices": [
        {
            "finish_reason": null,
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "全面"
            }
        }
    ],
    "usage": null
}
```
当接收到 Dify API 的 `message_end` 事件时
```json
{
    "id": "de80f5a9-6084-4324-9f78-92b51a91195b",
    "conversation_id": "0509851c-c6f3-4ebb-8319-cc045782c9fd",
    "task_id": "7fbb4a3f-079c-454f-a4cb-bca65adb60ca",
    "object": "chat.completion.usage",
    "created": 1720233867,
    "model": "dify",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 34,
            "delta": {
                "role": null,
                "content": null
            }
        }
    ],
    "usage": {
        "prompt_tokens": 22,
        "completion_tokens": 60,
        "total_tokens": 82
    }
}
```
