"""LLM factory. Returns an OpenAI-compatible chat model wired to local proxy or OpenRouter."""

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()


class ProxyChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant tailored for local web-reverse-proxies (WebAI-to-API).
    
    Omits 'temperature' and native 'response_format' that reverse proxies
    wrapping web chat interfaces reject with HTTP 400.
    """

    def with_structured_output(self, schema: Any, **kwargs: Any):
        schema_json = schema.model_json_schema()

        def _inject_schema(messages: Any) -> Any:
            instruction = (
                f"\n\nCRITICAL: You must respond ONLY with a valid JSON object matching this schema:\n"
                f"```json\n{schema_json}\n```\n"
                f"Do not include any explanation or conversational text outside the ```json ... ``` block."
            )
            if isinstance(messages, list):
                new_msgs = list(messages)
                if not new_msgs:
                    return new_msgs
                last = new_msgs[-1]
                if isinstance(last, dict):
                    new_msgs[-1] = {
                        **last,
                        "content": str(last.get("content", "")) + instruction,
                    }
                elif isinstance(last, BaseMessage):
                    new_msgs[-1] = last.copy(
                        update={"content": str(last.content) + instruction}
                    )
                return new_msgs
            return str(messages) + instruction

        def _parse(response: Any):
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            clean_text = match.group(1).strip() if match else text
            return schema.model_validate_json(clean_text)

        return (
            RunnableLambda(_inject_schema)
            | self
            | RunnableLambda(_parse)
        )


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or "sk-local-proxy"
    )
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or "http://localhost:6969/v1"
    )
    model = (
        os.environ.get("MODEL_NAME")
        or os.environ.get("LLM_MODEL")
        or "gemini-flash"
    )

    is_local_proxy = "localhost" in base_url or "127.0.0.1" in base_url

    if is_local_proxy:
        return ProxyChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            request_timeout=120.0,
            max_retries=2,
        )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
