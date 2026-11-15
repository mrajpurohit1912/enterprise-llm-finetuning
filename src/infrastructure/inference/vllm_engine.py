"""
src/infrastructure/inference/vllm_engine.py
vLLM production inference engine adapter optimized for local GPUs (e.g. RTX 4050 6GB).
"""

import time
import uuid
from collections.abc import AsyncIterator

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from src.domain.exceptions import ModelInferenceError
from src.domain.interfaces.inference_engine import IInferenceEngine
from src.domain.schemas.inference_schema import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)


class VLLMInferenceEngine(IInferenceEngine):
    """
    High-performance vLLM inference adapter implementing IInferenceEngine.
    Configured specifically for consumer GPUs (PagedAttention, 4-bit quantization, eager mode).
    """

    def __init__(
        self,
        model_path: str,
        quantization: str | None = "bitsandbytes",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.80,
        enable_lora: bool = True,
        max_lora_rank: int = 64,
    ) -> None:
        self.model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

        engine_args = AsyncEngineArgs(
            model=model_path,
            quantization=quantization,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=True,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank,
            trust_remote_code=False,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    @property
    def engine_name(self) -> str:
        return "vllm"

    def _convert_messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        return self._tokenizer.apply_chat_template(
            formatted,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _create_sampling_params(self, request: ChatCompletionRequest) -> SamplingParams:
        return SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt = self._convert_messages_to_prompt(request.messages)
        sampling_params = self._create_sampling_params(request)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        final_output = None
        async for output in self._engine.generate(prompt, sampling_params, request_id):
            final_output = output

        if final_output is None or not final_output.outputs:
            raise ModelInferenceError("vLLM engine produced no output.")

        first_choice = final_output.outputs[0]
        generated_text = first_choice.text
        finish_reason = first_choice.finish_reason or "stop"

        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(first_choice.token_ids)

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream_generate(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        prompt = self._convert_messages_to_prompt(request.messages)
        sampling_params = self._create_sampling_params(request)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        previous_text = ""
        results_generator = self._engine.generate(prompt, sampling_params, request_id)

        async for request_output in results_generator:
            if not request_output.outputs:
                continue

            first_choice = request_output.outputs[0]
            current_text = first_choice.text
            delta_text = current_text[len(previous_text):]
            previous_text = current_text

            finish_reason = first_choice.finish_reason

            yield ChatCompletionChunk(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=delta_text),
                        finish_reason=finish_reason,
                    )
                ],
            )