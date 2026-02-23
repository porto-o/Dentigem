import os
from dotenv import load_dotenv
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

GOOGLE_PROVIDER = GoogleProvider(api_key=GOOGLE_API_KEY)
OLLAMA_PROVIDER = OllamaProvider(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

GEMINI_MODEL = GoogleModel('gemini-3.1-pro-preview', provider=GOOGLE_PROVIDER)
OLLAMA_MODEL = OpenAIChatModel('gpt-oss:120b-cloud', provider=OLLAMA_PROVIDER)

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MEDGEMMA_PROCESSOR = AutoProcessor.from_pretrained("google/medgemma-4b-it")
MEDGEMMA_MODEL = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-4b-it", torch_dtype=torch.bfloat16, device_map="cuda"
)