import os
import gradio as gr
import asyncio
import websockets
import json
import base64
import numpy as np

# Env Vars
VLLM_URL = os.getenv("VLLM_URL", "https://rhaiis-inference-demo.apps.prime.pitt.ca/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-4B-Realtime-2602")

# Global state to keep the websocket alive
state = {"ws": None, "transcript": ""}

async def get_connection():
    if state["ws"] is None:
        # Convert https://.../v1 to wss://.../v1/realtime
        ws_url = VLLM_URL.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/realtime"):
            ws_url = f"{ws_url.rstrip('/')}/realtime"
        
        state["ws"] = await websockets.connect(ws_url)
        # Initialize session
        await state["ws"].send(json.dumps({
            "type": "session.update",
            "session": {"modalities": ["text"], "instructions": "Translate audio to English."}
        }))
    return state["ws"]

async def translate_audio(audio):
    if audio is None: return state["transcript"]
    
    try:
        ws = await get_connection()
        sampling_rate, data = audio
        
        # Voxtral/vLLM expects PCM16 at 16kHz
        # Convert Gradio's float32/int32 to base64 PCM16
        audio_bytes = (data.astype(np.int16)).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Append audio to vLLM buffer
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        
        # Wait for a response (non-blocking)
        # In a full app, you'd run a listener loop; here we check for recent events
        resp = await asyncio.wait_for(ws.recv(), timeout=0.1)
        resp_data = json.loads(resp)
        
        if resp_data.get("type") == "response.audio_transcription.delta":
            state["transcript"] += resp_data.get("delta", "")
            
    except Exception as e:
        print(f"Error: {e}")
        state["ws"] = None # Reset connection on error
        
    return state["transcript"]

with gr.Blocks() as demo:
    gr.Markdown(f"### 🎙️ Voxtral Translator\n**Endpoint:** `{VLLM_URL}`")
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], streaming=True)
        output_text = gr.Textbox(label="Real-time Translation", placeholder="Speak to see translation...")

    audio_input.stream(fn=translate_audio, inputs=audio_input, outputs=output_text, show_progress="hidden")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)