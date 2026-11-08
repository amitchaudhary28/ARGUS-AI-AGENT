
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from backend/.env
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env", override=False)


# Read Gemini configuration
API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def main():
    if not API_KEY:
        raise RuntimeError(
            "Gemini API key not found. "
            "Check GOOGLE_API_KEY in backend/.env"
        )

    print("=" * 55)
    print("ARGUS - GEMINI CONNECTION TEST")
    print("=" * 55)

    print(f"Model requested: {MODEL}")
    print("Initializing Google GenAI SDK...\n")

    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            timeout=60000
        ),
    )

    start_time = time.perf_counter()

    try:
        print("Sending test request to Gemini...\n")

        response = client.models.generate_content(
            model=MODEL,
            contents="Reply with exactly: ARGUS_OK",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
            ),
        )

        elapsed = time.perf_counter() - start_time

        print("=" * 55)
        print("GEMINI RESPONSE")
        print("=" * 55)

        print("Requested model:", MODEL)
        print("Actual model:", response.model_version)
        print("Response ID:", response.response_id)
        print("Finish reason:",
              response.candidates[0].finish_reason
              if response.candidates else "No candidate")

        answer = response.text

        if answer:
            print("\nResponse:", answer)
            print("\nSTATUS: SUCCESS")
        else:
            print("\nNo text response was generated.")
            print("Raw response:", repr(response))

            if response.candidates:
                for candidate in response.candidates:
                    print("Candidate content:", candidate.content)
                    print("Finish reason:", candidate.finish_reason)

            print("\nSTATUS: EMPTY RESPONSE")

        print(f"\nResponse time: {elapsed:.2f} seconds")

        if response.usage_metadata:
            print(
                "Total tokens:",
                response.usage_metadata.total_token_count
            )

    except Exception as exc: # noqa: BLE001
        elapsed = time.perf_counter() - start_time

        print("\nSTATUS: FAILED")
        print("Exception:", type(exc).__name__)
        print("Error:", str(exc))
        print(f"Time before failure: {elapsed:.2f} seconds")

    finally:
        client.close()

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
