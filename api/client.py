from env import env
from httpx import AsyncClient

httpx_client = AsyncClient(
    base_url=env.str("DIFY_API_URL")
)