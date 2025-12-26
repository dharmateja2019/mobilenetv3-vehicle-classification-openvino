import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None


def _load_model():
    """Lazy-load Qwen model (CPU-safe)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME, trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        _model.eval()


def ask_question(vision_result: dict, question: str) -> str:
    _load_model()

    if not isinstance(vision_result, dict):
        return "Vision results are not available."

    vehicles = vision_result.get("vehicles", [])

    if not vehicles:
        context = "No vehicles were detected in the image."
    else:
        context = "Detected vehicles:\n"
        for idx, v in enumerate(vehicles, start=1):
            v_type = v.get("type", "unknown")
            v_color = v.get("color", "unknown")
            context += (
                f"- Vehicle {idx}: Type = {v_type} (vehicle class), "
                f"Color = {v_color} (paint color)\n"
            )


    prompt = f"""
You are an assistant analyzing vehicle detection results.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    inputs = _tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    return _tokenizer.decode(output[0], skip_special_tokens=True)
