import os
import json
import base64
import asyncio
import logging
import numpy as np
import gradio as gr
import websockets
from scipy.signal import resample

# Configure logging for OpenShift pod logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoxtralApp")

# Configuration from OpenShift Env Vars
VLLM_URL = os.getenv("VLLM_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-4B-Realtime-2602")

class VoxtralClient:
    def __init__(self):
        self.ws = None
        self.transcript = ""

    async def connect(self):
        """Establish WebSocket connection with protocol swap and vLLM schema."""
        ws_url = VLLM_URL.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/realtime"):
            ws_url = f"{ws_url.rstrip('/')}/realtime"
        
        logger.info(f"Connecting to: {ws_url}")
        self.ws = await websockets.connect(ws_url, ping_interval=20)
        
        # FIX: vLLM expects 'model' at the top level, not inside 'session'
        init_event = {
            "type": "session.update",
            "model": MODEL_NAME,
            "session": {
                "modalities": ["text"],
                "instructions": "Translate the following audio to English clearly and concisely.",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }
        await self.ws.send(json.dumps(init_event))
        logger.info("Session initialized successfully.")

    async def stream_audio(self, audio):
        """Process chunks: Resample -> Encode -> Send -> Receive."""
        if audio is None:
            return self.transcript

        try:
            if self.ws is None or self.ws.closed:
                await self.connect()

            sr, data = audio
            
            # 1. Resample to 16kHz (Voxtral Standard)
            target_sr = 16000
            if sr != target_sr:
                num_samples = int(len(data) * target_sr / sr)
                data = resample(data, num_samples)
            
            # 2. Convert to PCM16 and Base64
            audio_pcm16 = data.astype(np.int16).tobytes()
            audio_b64 = base64.b64encode(audio_pcm16).decode("utf-8")

            # 3. Send to vLLM
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }))

            # 4. Listen for transcription/translation deltas
            # We use a short timeout so we don't block the next audio chunk
            try:
                while True:
                    raw_resp = await asyncio.wait_for(self.ws.recv(), timeout=0.05)
                    resp = json.loads(raw_resp)
                    
                    if resp.get("type") == "response.audio_transcription.delta":
                        self.transcript += resp.get("delta", "")
                    elif resp.get("type") == "error":
                        logger.error(f"vLLM Error: {resp.get('error')}")
            except asyncio.TimeoutError:
                pass # No more messages for this chunk

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.ws = None # Trigger reconnect on next chunk
            
        return self.transcript

# Persistent client state
client_instance = VoxtralClient()

# Gradio Interface
with gr.Blocks(title="Voxtral Real-time Translator") as demo:
    gr.Markdown(f"## 🎙️ Voxtral Real-time Gateway\n**Backend:** `{VLLM_URL}`")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"], 
                streaming=True, 
                label="Speak Here"
            )
        with gr.Column():
            output_text = gr.Textbox(
                label="Live Translation (English)", 
                interactive=False,
                lines=10
            )

    # Audio stream triggers the async translation function
    audio_input.stream(
        fn=client_instance.stream_audio, 
        inputs=[audio_input], 
        outputs=[output_text],
        show_progress="hidden"
    )

if __name__ == "__main__":
    # OpenShift default port 8080
    demo.launch(server_name="0.0.0.0", server_port=8080)