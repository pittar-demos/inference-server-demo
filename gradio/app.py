import os
import json
import base64
import asyncio
import logging
import numpy as np
import gradio as gr
import websockets
from scipy.signal import resample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoxtralApp")

VLLM_URL = os.getenv("VLLM_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-4B-Realtime-2602")

if not VLLM_URL:
    raise RuntimeError("VLLM_URL environment variable is not set!")

class VoxtralClient:
    def __init__(self):
        self.ws = None
        self.transcript = ""
        self.lock = asyncio.Lock()

    def is_connected(self):
        """Version-agnostic check for websocket state."""
        if self.ws is None:
            return False
        # Modern websockets use .open; older ones used .closed
        return getattr(self.ws, "open", not getattr(self.ws, "closed", True))

    async def connect(self):
        ws_url = VLLM_URL.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/realtime"):
            ws_url = f"{ws_url.rstrip('/')}/realtime"
        
        logger.info(f"Connecting to: {ws_url}")
        self.ws = await websockets.connect(ws_url, ping_interval=20)
        
        init_event = {
            "type": "session.update",
            "model": MODEL_NAME,
            "session": {
                "modalities": ["text"],
                "instructions": "Translate all incoming audio to English. Be concise.",
                "input_audio_format": "pcm16"
            }
        }
        await self.ws.send(json.dumps(init_event))
        logger.info("Connection established and session updated.")

    async def stream_audio(self, audio):
        if audio is None:
            return self.transcript

        async with self.lock:
            try:
                if not self.is_connected():
                    await self.connect()

                sr, data = audio
                # Resample to 16kHz PCM16
                if sr != 16000:
                    data = resample(data, int(len(data) * 16000 / sr))
                
                audio_b64 = base64.b64encode(data.astype(np.int16).tobytes()).decode("utf-8")

                await self.ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))

                # Non-blocking check for responses
                try:
                    while True:
                        raw = await asyncio.wait_for(self.ws.recv(), timeout=0.02)
                        resp = json.loads(raw)
                        # vLLM supports both transcription.delta and response.text.delta
                        # depending on the specific Voxtral sub-version
                        if resp.get("type") in ["response.text.delta", "transcription.delta"]:
                            self.transcript += resp.get("delta", "")
                        elif resp.get("type") == "error":
                            logger.error(f"vLLM Error: {resp.get('error')}")
                except asyncio.TimeoutError:
                    pass 

            except Exception as e:
                logger.error(f"Critical Streaming Error: {e}")
                self.ws = None # Reset for next attempt
            
        return self.transcript

    def reset(self):
        self.transcript = ""
        return ""

client = VoxtralClient()

with gr.Blocks(title="Voxtral Translator", css="footer {visibility: hidden}") as demo:
    gr.Markdown(f"### 🌍 Voxtral Real-time Translator\n**Endpoint:** `{VLLM_URL}`")
    
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], streaming=True, type="numpy")
        text_out = gr.Textbox(label="English Translation", lines=10, interactive=False)
    
    with gr.Row():
        clear_btn = gr.Button("🗑️ Reset Transcript", variant="stop")

    # The stream event
    audio_in.stream(
        fn=client.stream_audio, 
        inputs=[audio_in], 
        outputs=[text_out], 
        show_progress="hidden",
        queue=True # Essential for non-blocking UI
    )
    
    # Explicitly handle stop to prevent browser hang
    audio_in.stop_recording(fn=lambda: None)
    
    clear_btn.click(fn=client.reset, outputs=[text_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)