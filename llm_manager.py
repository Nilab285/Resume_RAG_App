import random
import time
from threading import BoundedSemaphore

from openai import OpenAI

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    MAX_CONCURRENT_LLM,
    LLM_MAX_RETRIES,
    LLM_INITIAL_BACKOFF,
)

# ----------------------------
# Singleton Client
# ----------------------------

_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# ----------------------------
# Concurrency Limiter
# ----------------------------

_llm_semaphore = BoundedSemaphore(MAX_CONCURRENT_LLM)


def llm_chat(
    messages,
    model=LLM_MODEL,
    temperature=0.0,
    **kwargs,
):
    """
    Thread-safe wrapper around Groq.

    Features:
    - Limits concurrent requests
    - Retries on transient failures
    - Exponential backoff
    - Jitter
    """

    with _llm_semaphore:

        delay = LLM_INITIAL_BACKOFF

        for attempt in range(LLM_MAX_RETRIES):

            try:

                return _client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )

            except Exception as e:

                message = str(e).lower()

                retryable = (
                    "429" in message
                    or "rate" in message
                    or "timeout" in message
                    or "connection" in message
                    or "temporarily" in message
                )

                if not retryable:
                    raise

                if attempt == LLM_MAX_RETRIES - 1:
                    raise

                sleep_time = delay + random.uniform(0, 0.5)

                print(
                    f"[LLM] Retry {attempt + 1}/{LLM_MAX_RETRIES} "
                    f"in {sleep_time:.2f}s"
                )

                time.sleep(sleep_time)

                delay *= 2