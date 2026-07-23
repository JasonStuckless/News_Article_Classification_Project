"""
ollama_client.py

Client utilities for communicating with a locally running Ollama server.

This module is responsible for:
- Connecting to Ollama.
- Verifying that the selected model is available.
- Sending prompts to the model.
- Requesting structured JSON output.
- Retrying temporary connection or server failures.
- Returning the raw response and inference metadata.

Label validation is handled separately in validate_labels.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import httpx
import ollama
from ollama import Client

from utils.logging import get_logger


logger = get_logger(__name__)


class OllamaClientError(RuntimeError):
    """Raised when a request to Ollama cannot be completed."""


@dataclass(frozen=True)
class OllamaResponse:
    """
    Response returned by the local Ollama model.

    Attributes
    ----------
    content:
        Raw text returned in the assistant message.

    model:
        Name of the model that produced the response.

    total_duration_ns:
        Total Ollama request duration in nanoseconds, when available.

    prompt_eval_count:
        Number of input tokens processed, when available.

    eval_count:
        Number of output tokens generated, when available.
    """

    content: str
    model: str
    total_duration_ns: int | None
    prompt_eval_count: int | None
    eval_count: int | None


class OllamaClient:
    """
    Client for sending classification prompts to a local Ollama model.

    Parameters
    ----------
    model:
        Ollama model name, such as ``deepseek-r1:8b``.

    host:
        Address of the local Ollama server.

    temperature:
        Sampling temperature used during inference. A value of 0.0 is
        recommended for consistent classification output.

    timeout_seconds:
        Maximum time allowed for one request.

    max_retries:
        Number of additional attempts after the first failed request.

    retry_delay_seconds:
        Initial delay between retries. The delay increases after each
        failed attempt.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 2.0,
    ) -> None:
        if not model.strip():
            raise ValueError("The Ollama model name cannot be empty.")

        if temperature < 0:
            raise ValueError("Temperature cannot be negative.")

        if timeout_seconds <= 0:
            raise ValueError("Timeout must be greater than zero.")

        if max_retries < 0:
            raise ValueError("Maximum retries cannot be negative.")

        if retry_delay_seconds < 0:
            raise ValueError("Retry delay cannot be negative.")

        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        self._client = Client(
            host=self.host,
            timeout=timeout_seconds,
        )

    def verify_connection(self) -> None:
        """
        Verify that the Ollama server is running and reachable.

        Raises
        ------
        OllamaClientError
            If the local Ollama server cannot be reached.
        """
        try:
            self._client.list()
            logger.info("Connected to Ollama at %s.", self.host)

        except httpx.RequestError as error:
            raise OllamaClientError(
                f"Could not connect to Ollama at {self.host}. "
                "Confirm that the Ollama application is installed and running."
            ) from error

        except ollama.ResponseError as error:
            raise OllamaClientError(
                f"Ollama returned an error while checking the connection: "
                f"{error.error}"
            ) from error

    def verify_model(self) -> None:
        """
        Verify that the configured model is available locally.

        Raises
        ------
        OllamaClientError
            If the model is unavailable or Ollama cannot be reached.
        """
        try:
            self._client.show(self.model)
            logger.info("Ollama model is available: %s", self.model)

        except httpx.RequestError as error:
            raise OllamaClientError(
                f"Could not connect to Ollama at {self.host}."
            ) from error

        except ollama.ResponseError as error:
            if error.status_code == 404:
                raise OllamaClientError(
                    f"The Ollama model '{self.model}' is not installed. "
                    f"Install it with: ollama pull {self.model}"
                ) from error

            raise OllamaClientError(
                f"Ollama could not access model '{self.model}': "
                f"{error.error}"
            ) from error

    def generate(
        self,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> OllamaResponse:
        """
        Send a prompt to the configured Ollama model.

        Parameters
        ----------
        prompt:
            Complete prompt to send to the model.

        response_schema:
            Optional JSON schema used to constrain the response. If no schema
            is provided, Ollama is still instructed to return valid JSON.

        Returns
        -------
        OllamaResponse
            Raw response content and available inference metadata.

        Raises
        ------
        ValueError
            If the prompt is empty.

        OllamaClientError
            If the request fails after all retry attempts.
        """
        if not prompt.strip():
            raise ValueError("The prompt cannot be empty.")

        output_format: str | dict[str, Any]

        if response_schema is None:
            output_format = "json"
        else:
            output_format = response_schema

        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                logger.debug(
                    "Sending request to model %s. Attempt %d of %d.",
                    self.model,
                    attempt,
                    attempts,
                )

                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    stream=False,
                    format=output_format,
                    think=False,
                    options={
                        "temperature": self.temperature,
                    },
                )

                content = response.message.content

                if not content or not content.strip():
                    raise OllamaClientError(
                        "Ollama returned an empty response."
                    )

                return OllamaResponse(
                    content=content.strip(),
                    model=response.model or self.model,
                    total_duration_ns=response.total_duration,
                    prompt_eval_count=response.prompt_eval_count,
                    eval_count=response.eval_count,
                )

            except httpx.RequestError as error:
                last_error = error
                logger.warning(
                    "Could not connect to Ollama on attempt %d of %d: %s",
                    attempt,
                    attempts,
                    error,
                )

            except ollama.ResponseError as error:
                last_error = error
                logger.warning(
                    "Ollama request failed on attempt %d of %d: %s",
                    attempt,
                    attempts,
                    error.error,
                )

                # A missing model will not be fixed by retrying.
                if error.status_code == 404:
                    raise OllamaClientError(
                        f"The Ollama model '{self.model}' is unavailable. "
                        f"Install it with: ollama pull {self.model}"
                    ) from error

            except OllamaClientError as error:
                last_error = error
                logger.warning(
                    "Invalid Ollama response on attempt %d of %d: %s",
                    attempt,
                    attempts,
                    error,
                )

            if attempt < attempts:
                delay = self.retry_delay_seconds * attempt

                logger.info(
                    "Retrying Ollama request in %.1f seconds.",
                    delay,
                )

                time.sleep(delay)

        raise OllamaClientError(
            f"Ollama request failed after {attempts} attempts."
        ) from last_error

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> OllamaClient:
        """Return the client when used as a context manager."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> None:
        """Close the client when leaving a context manager."""
        self.close()