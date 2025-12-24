# src/vlm_reasoner.py

import base64
import json
import os
import time
import streamlit as st
from openai import OpenAI, RateLimitError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@st.cache_data(show_spinner=False)
def _vlm_cached(image_path: str):
    start = time.time()
    image_base64 = _encode_image(image_path)

    prompt = """
You are a vehicle analysis assistant.

Look at the image and respond STRICTLY in JSON:

{
  "type": "car | motorcycle | scooter | bus | truck | unknown",
  "color": "single word color",
  "description": "short description"
}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=120,
        temperature=0.2,
    )

    latency_ms = round((time.time() - start) * 1000, 2)

    content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {
            "type": "unknown",
            "color": "unknown",
            "description": content,
        }

    parsed["latency_ms"] = latency_ms
    return parsed


def vlm_describe(image_path: str):
    try:
        return _vlm_cached(image_path)

    except RateLimitError:
        return {
            "type": "unknown",
            "color": "unknown",
            "description": "VLM skipped due to rate limit",
            "latency_ms": None,
        }

    except Exception as e:
        return {
            "type": "unknown",
            "color": "unknown",
            "description": f"VLM error: {str(e)}",
            "latency_ms": None,
        }
