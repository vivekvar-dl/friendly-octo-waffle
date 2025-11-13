from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
DEFAULT_LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

app = FastAPI(
    title="Qwen LLM Service",
    version="0.1.0",
    description="Lightweight inference endpoint that serves Qwen/Qwen3-8B on a single GPU.",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to send to the language model.")
    max_new_tokens: Optional[int] = Field(
        None,
        gt=0,
        le=2048,
        description="Maximum number of tokens to generate. Defaults to MAX_NEW_TOKENS env or 512.",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Defaults to TEMPERATURE env or 0.1.",
    )
    top_p: Optional[float] = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling value.",
    )


class GenerateResponse(BaseModel):
    completion: str


@lru_cache(maxsize=1)
def _load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=1)
def _load_model():
    quantization_config = None
    model_kwargs: dict[str, object] = {"trust_remote_code": True}

    if DEFAULT_LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        device_map="auto" if torch.cuda.is_available() else None,
        **model_kwargs,
    )

    if tokenizer := _load_tokenizer():
        # ensure pad token id aligns with eos for generation
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    return model


@app.get("/healthz")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_text(payload: GenerateRequest) -> GenerateResponse:
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must be a non-empty string.")

    tokenizer = _load_tokenizer()
    model = _load_model()

    max_new_tokens = payload.max_new_tokens or DEFAULT_MAX_NEW_TOKENS
    temperature = payload.temperature if payload.temperature is not None else DEFAULT_TEMPERATURE

    do_sample = temperature > 0
    inputs = tokenizer(
        payload.prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=payload.top_p or 0.95,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware specific
        raise HTTPException(
            status_code=503,
            detail="CUDA out of memory. Try reducing max_new_tokens or temperature.",
        ) from exc

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated_text[len(payload.prompt) :].strip()

    return GenerateResponse(completion=completion or generated_text.strip())

