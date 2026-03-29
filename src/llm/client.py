"""Ollama 本地 LLM 客户端封装。"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import ollama

_LOG = logging.getLogger(__name__)


class OllamaClient:
    """轻量 Ollama 客户端，支持重试与结构化输出解析。"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 3.0,
        base_url: str = "http://localhost:11434",
        ollama_options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # 与每次请求合并，用于 num_ctx / num_gpu / num_thread 等，以吃满本地 GPU 或控制显存
        self._ollama_options: dict[str, Any] = dict(ollama_options) if ollama_options else {}
        self._client = ollama.Client(host=base_url, timeout=timeout)

    def _merged_options(self, temperature: float) -> dict[str, Any]:
        out = {**self._ollama_options, "temperature": temperature}
        return out

    # ------------------------------------------------------------------
    # 基础调用
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        """发送对话，返回模型回复文本；失败时重试。"""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options=self._merged_options(temperature),
                )
                return resp["message"]["content"].strip()
            except Exception as exc:
                last_exc = exc
                _LOG.warning("Ollama chat 失败 (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"Ollama 调用失败（{self.max_retries + 1} 次）：{last_exc}") from last_exc

    def chat_json(self, messages: list[dict], temperature: float = 0.1) -> dict[str, Any]:
        """调用模型，期望返回 JSON，自动解析；解析失败返回空 dict。"""
        raw = self.chat(messages, temperature=temperature)
        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 提取 ```json ... ``` 代码块
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # 提取第一个 { ... }
        m2 = re.search(r"\{.*\}", raw, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except json.JSONDecodeError:
                pass
        _LOG.warning("LLM 返回内容无法解析为 JSON，原始内容：%s", raw[:200])
        return {}

    def is_available(self) -> bool:
        """检查 Ollama 服务是否可达且模型已加载。"""
        try:
            models = self._client.list()
            names = [m["model"] for m in models.get("models", [])]
            return any(self.model in n for n in names)
        except Exception:
            return False
