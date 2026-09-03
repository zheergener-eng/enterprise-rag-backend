"""DeepSeek LLM 客户端封装（OpenAI 兼容协议）。

职责单一：给定 prompt，调用 deepseek-chat 并返回回答文本。

- 非流式：`generate(prompt) -> str`。
- 流式：`stream_generate(prompt)`，逐步 yield 非空文本增量。

Retrieval / RAG Prompt / Query Rewrite / Session / StreamingResponse 均不在此层，
由上层（RAG 服务 / API 层）负责。二者共用同一 OpenAI client（懒加载单例），
不因流式而为每个 token 新建连接。
"""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import settings


class LLMError(Exception):
    """LLM 调用相关错误（配置缺失 / 调用失败 / 空响应）。"""


class DeepSeekClient:
    """基于 OpenAI 兼容 SDK 的 DeepSeek 非流式客户端。"""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """懒加载 OpenAI 客户端（进程内复用同一连接）。"""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, prompt: str) -> str:
        """调用 DeepSeek 生成回答（非流式）。

        Args:
            prompt: 用户 prompt 文本。

        Returns:
            模型生成的回答文本（非空）。

        Raises:
            LLMError: API Key 缺失、API 调用失败、或模型返回空内容。
        """
        if not self.api_key:
            raise LLMError(
                "DEEPSEEK_API_KEY is not configured; "
                "set it in .env (see .env.example) before calling the LLM"
            )

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # 网络 / 鉴权 / 服务端错误等
            raise LLMError(f"DeepSeek API call failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMError("DeepSeek returned empty content")

        return content

    def stream_generate(self, prompt: str):
        """调用 DeepSeek 生成回答（流式），逐步 yield 非空文本增量。

        职责边界：仅 prompt → DeepSeek stream → 文本增量；
        不含 Session / Retrieval / Query Rewrite / Redis / StreamingResponse。

        Args:
            prompt: 用户 prompt 文本。

        Yields:
            非空文本增量（str）。跳过无 choices 的 chunk、delta.content 为 None
            或空字符串的 chunk。

        Raises:
            LLMError: API Key 缺失、建立流失败（首个 token 前）、或流中途失败。
                错误消息区分“首个 token 前”与“中途”两个阶段，供上层判断
                失败发生在产生任何输出之前还是之后。

        注意：本方法为 generator（含 yield），函数体在首次迭代时才执行，
        因此 API Key 校验与连接建立也延迟到迭代时触发。通过 `with` 管理流对象，
        正常结束或异常时都能正确释放底层 HTTP 连接。
        """
        if not self.api_key:
            raise LLMError(
                "DEEPSEEK_API_KEY is not configured; "
                "set it in .env (see .env.example) before calling the LLM"
            )

        client = self._get_client()
        started = False  # 是否已产出过至少一个 token（用于区分失败阶段）
        try:
            with client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ) as stream:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if not content:
                        continue
                    started = True
                    yield content
        except Exception as exc:  # 网络 / 鉴权 / 服务端 / 迭代中途错误等
            phase = "mid-generation" if started else "before first token"
            raise LLMError(f"DeepSeek stream failed {phase}: {exc}") from exc


@lru_cache
def get_llm_client() -> DeepSeekClient:
    """返回进程级单例 DeepSeek 客户端。"""
    return DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
    )


# 便捷单例：`from app.services.llm import llm_client`
llm_client = get_llm_client()
