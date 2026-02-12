import os
import gradio as gr
from openai import OpenAI

# Configuration via Environment Variables
VLLM_URL = os.getenv("VLLM_URL", "http://your-remote-vllm-ip:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-4B-Realtime-2602")

# Initialize Client
client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")

def translate_audio(audio_path):
    if not audio_path:
        return "No audio recorded."
    
    # Voxtral handles transcription and translation natively
    # Using the standard OpenAI-compatible audio transcriptions endpoint
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=MODEL_NAME, 
            file=audio_file,
            response_format="text"
        )
    return transcription

demo = gr.Interface(
    fn=translate_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text",
    title="Voxtral Real-time Translator",
    description=f"Connected to vLLM at: {VLLM_URL}"
)

if __name__ == "__main__":
    # OpenShift requires listening on 0.0.0.0 and a non-privileged port (8080)
    demo.launch(server_name="0.0.0.0", server_port=8080)