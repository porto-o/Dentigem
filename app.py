import gradio as gr
from denti_core.utils import _read_patient_data
from denti_core.assistant_agent import AssistantDeps, assistant


def select_patient(patient_path: str):
    content = _read_patient_data(patient_path)
    history = [{"role": "assistant", "content": [{"type": "text", "text": "Patient records loaded, ask me anything! 🦷"}]}]
    return history, content, []

def respond(message, _history, patient_context, agent_history):
    query = message.get("text", "")
    files = message.get("files", [])

    if not patient_context:
        return "Select a patient first."

    deps = AssistantDeps(patient_context=patient_context, image_path=files[0] if files else None)
    result = assistant.run_sync(query, deps=deps, message_history=agent_history)
    agent_history.extend(result.new_messages())

    return result.output


with gr.Blocks(title="DentalGemma", theme=gr.themes.Soft()) as dentigem:

    patient_history = gr.State()
    agent_history = gr.State([])

    gr.Markdown("# 🦷 DentalGemma")

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            patient = gr.FileExplorer(
                label="Patient Records",
                file_count="single",
                glob="*.md",
                root_dir="./DentiGem_Demo/Patients",
                interactive=True,
                height=600,
            )
            btn_select = gr.Button(value="Load Patient", variant="primary")

        with gr.Column(scale=3):
            chat_window = gr.Chatbot(
                label="DentalGemma Assistant",
                placeholder="<strong>Select a patient record to begin.</strong>",
                height=600
            )
            gr.ChatInterface(
                fn=respond,
                multimodal=True,
                autoscroll=True,
                chatbot=chat_window,
                additional_inputs=[patient_history, agent_history],
            )

    btn_select.click(
        fn=select_patient, inputs=patient, outputs=[chat_window, patient_history, agent_history]
    )

dentigem.launch(theme=gr.themes.Soft())