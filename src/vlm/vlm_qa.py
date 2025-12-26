from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

_processor = None
_model = None

def _load_model():
    global _processor, _model
    if _model is None:
        _processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        _model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        _model.eval()

def ask_question(image, question: str) -> str:
    _load_model()   # <-- loads ONLY when question is asked

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")

    inputs = _processor(image, question, return_tensors="pt")

    with torch.no_grad():
        output = _model.generate(**inputs, max_new_tokens=20)

    return _processor.decode(output[0], skip_special_tokens=True)
