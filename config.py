"""Configuración central del agente.

Lección 1: principio fail-fast.
Si falta una variable en .env, fallamos AQUÍ con un mensaje claro,
en lugar de fallar después dentro de OpenAI/LangChain con un error críptico.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(
            f"Falta la variable de entorno {name}. "
            f"Revísala en tu archivo .env "
            f"(ejemplo: {name}=tu_valor). "
            f"Variables requeridas: API_URL, API_KEY, AI_MODEL."
        )
    return value.strip()


API_URL: str = _require_env("API_URL")
API_KEY: str = _require_env("API_KEY")
# AI_MODEL se lee del .env para que puedas probar varios modelos
# free de OpenRouter sin tocar código. Ej: nvidia/nemotron-3-super-120b-a12b:free
AI_MODEL: str = _require_env("AI_MODEL")
