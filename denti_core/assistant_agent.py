from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from denti_core.load_models import GEMINI_MODEL, OLLAMA_MODEL
from denti_core.medgemma import analyze_xray

@dataclass
class AssistantDeps:
    patient_context: str
    image_path: str | None = None

assistant = Agent(
    name="DentalGemma Agent",
    model=OLLAMA_MODEL,
    deps_type=AssistantDeps,
    system_prompt="""You are DentalGemma, a dental assistant AI. Answer questions based on the patient records provided.
    IMPORTANT: When the user sends an image or mentions an X-ray or radiography, you MUST call the analyze_xray_tool. Never say you cannot view images.
    Keep your responses concise but not too ambiguous"""
)

@assistant.system_prompt
def inject_patient_context(ctx: RunContext[AssistantDeps]) -> str:
    return f"Current patient record:\n\n{ctx.deps.patient_context}"


@assistant.tool
async def analyze_xray_tool(ctx: RunContext[AssistantDeps]) -> str:
    """Analyze a dental X-ray or radiography image. Call this tool when the user sends an image."""
    print(f'TOOL: {ctx.deps.image_path}')
    if not ctx.deps.image_path:
        return "No image was provided."
    result = await analyze_xray(ctx.deps.image_path)
    print(f"MEDGEMMA RESULT: {result}")
    return result