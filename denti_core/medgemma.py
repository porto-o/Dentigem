from denti_core.load_models import MEDGEMMA_MODEL, MEDGEMMA_PROCESSOR
from PIL import Image


async def analyze_xray(image_path: str) -> str:
    image = Image.open(image_path)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are an expert dental radiologist. When analyzing a dental X-ray, structure your response in three sections:

                    TECHNICAL: Describe findings for the dentist — bone levels, pathologies, restorations, root morphology, periapical status, any anomalies.
                    PATIENT: Explain the same findings in simple, calm language a patient can understand. No jargon.
                    RECOMMENDATIONS: Suggest procedures, treatments, or medications based on your findings.

                    Be thorough but concise. If findings are limited, reflect that honestly.
                    """,
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please analyze this dental X-ray."},
                {"type": "image", "image": image},
            ],
        },
    ]
    inputs = MEDGEMMA_PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    output = MEDGEMMA_MODEL.generate(**inputs, max_new_tokens=64)
    decoded = MEDGEMMA_PROCESSOR.decode(output[0], skip_special_tokens=True)
    print(f"DECODED: {decoded}")
    return decoded
