import os
import runpod
from functionary.vllm_inference import process_chat_completion
from functionary.openai_types import ChatCompletionRequest
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio

# Retrieve the model name from environment variables
model_name = os.getenv("MODEL_NAME", "default_model_name")

# Initialize the engine and tokenizer using the environment variable
engine_args = AsyncEngineArgs(model=model_name)
engine = AsyncLLMEngine.from_engine_args(engine_args)
tokenizer = get_tokenizer(
    engine_args.tokenizer, tokenizer_mode=engine_args.tokenizer_mode
)
engine_model_config = asyncio.run(engine.get_model_config())


async def handler(job):
    job_input = job["input"]
    request = ChatCompletionRequest(**job_input)

    response = await process_chat_completion(
        request=request,
        raw_request=None,
        tokenizer=tokenizer,
        served_model=model_name,
        engine_model_config=engine_model_config,
        enable_grammar_sampling=False,
        engine=engine,
    )

    return response


runpod.serverless.start({"handler": handler})
