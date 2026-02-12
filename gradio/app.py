import os
import gradio as gr
import httpx
import asyncio
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoxtralDebug")

VLLM_URL = os.getenv("VLLM_URL", "").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-4B-Realtime-2602")

def check_vllm_health():
    """Verify vLLM is reachable at startup."""
    try:
        # vLLM provides a /health or /v1/models endpoint
        resp = httpx.get(f"{VLLM_URL}/models", timeout=5.0)
        if resp.status_code == 200:
            return f"✅ Connected to vLLM! Model: {MODEL_NAME}"
        return f"❌ vLLM returned status {resp.status_code}"
    except Exception as e:
        return f"❌ Connection Failed: {str(e)}"

async def translate_stream(audio):
    """Handles the incoming audio stream chunks."""
    if audio is None:
        return ""
    
    # DEBUG: Log that we are receiving audio
    logger.info(f"Received audio chunk: {len(audio[1])} samples")
    
    # For a real implementation, you would use 'websockets' library here
    # to pipe 'audio' to vLLM. For now, let's just confirm the UI works:
    return "Audio chunk received... processing..."

with gr.Blocks() as demo:
    status_label = gr.Markdown(check_vllm_health())
    
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], streaming=True)
        text_out = gr.Textbox(label="Live Translation")
    
    audio_in.stream(fn=translate_stream, inputs=[audio_in], outputs=[text_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)