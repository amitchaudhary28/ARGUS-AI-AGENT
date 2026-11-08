
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("Gemini API key not found.")

client = genai.Client(api_key=api_key)

print("Checking models available to ARGUS API key...\n")

try:
    for model in client.models.list():
        name = getattr(model, "name", "")
        actions = getattr(model, "supported_actions", None)

        if "gemini" in str(name).lower():
            print("Model:", name)
            print("Supported actions:", actions)
            print("-" * 50)

except Exception as exc: # noqa: BLE001
    print("Model listing failed:")
    print(type(exc).__name__, str(exc))

finally:
    client.close()
