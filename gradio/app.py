import os, json, base64, asyncio, logging
import numpy as np
import gradio as gr
import websockets
from scipy.signal import resample
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoxtralApp")

VLLM_URL = os.getenv("VLLM_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-4B-Realtime-2602")

class VoxtralClient:
    def __init__(self):
        self.ws = None
        self.transcript = ""
        self.lock = asyncio.Lock()

    async def connect(self):
        ws_url = VLLM_URL.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/realtime"):
            ws_url = f"{ws_url.rstrip('/')}/realtime"

        # Increased timeout and ping for OpenShift stability
        self.ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=20)

        init_event = {
            "type": "session.update",
            "model": MODEL_NAME,
            "session": {
                "modalities": ["text"],
                "instructions": "Translate audio to English. Output only the translation.",
                "input_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"} # Forces vLLM to detect when you stop talking
            }
        }
        await self.ws.send(json.dumps(init_event))
        logger.info("Connection established with Server VAD enabled.")

    async def stream_audio(self, audio):
        if audio is None:
            return self.transcript

        async with self.lock:
            try:
                if self.ws is None or not getattr(self.ws, "open", False):
                    await self.connect()

                sr, data = audio
                # Strict 16kHz conversion
                if sr != 16000:
                    data = resample(data, int(len(data) * 16000 / sr))

                audio_b64 = base64.b64encode(data.astype(np.int16).tobytes()).decode("utf-8")

                # Send audio chunk
                await self.ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))

                # Try to catch any incoming text deltas
                try:
                    while True:
                        # Ultra-short timeout to prevent UI lag
                        raw = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                        resp = json.loads(raw)
                        t = resp.get("type")

                        # Catching all possible text keys in vLLM/Voxtral
                        if t in ["response.text.delta", "response.audio_transcription.delta"]:
                            delta = resp.get("delta", "") or resp.get("transcript", "")
                            self.transcript += delta
                        elif t == "error":
                            logger.error(f"vLLM Error: {resp.get('error')}")
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Stream error: {e}")
                self.ws = None

        return self.transcript

    async def process_file(self, file_path):
        """Process an uploaded audio file for translation."""
        if file_path is None:
            return "Please upload a file."

        try:
            logger.info(f"Processing file: {file_path}")

            # Load audio file with librosa (handles most audio formats)
            # Automatically resamples to 16kHz and converts to mono
            audio_data, sr = librosa.load(file_path, sr=16000, mono=True)

            # Convert float32 [-1, 1] to int16 PCM16 format
            audio_data = (audio_data * 32767).astype(np.int16)

            # Connect to WebSocket
            async with self.lock:
                if self.ws is None or not getattr(self.ws, "open", False):
                    await self.connect()

                # Send audio in chunks (1 second chunks for better responsiveness)
                chunk_size = 16000  # 1 second at 16kHz
                file_transcript = ""

                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    audio_b64 = base64.b64encode(chunk.tobytes()).decode("utf-8")

                    await self.ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }))

                    # Collect responses for this chunk
                    try:
                        while True:
                            raw = await asyncio.wait_for(self.ws.recv(), timeout=0.05)
                            resp = json.loads(raw)

                            if resp.get("type") in ["response.text.delta", "transcription.delta"]:
                                file_transcript += resp.get("delta", "")
                            elif resp.get("type") == "error":
                                logger.error(f"vLLM Error: {resp.get('error')}")
                                return f"Error: {resp.get('error')}"
                    except asyncio.TimeoutError:
                        pass

                # Wait a bit for final responses
                try:
                    while True:
                        raw = await asyncio.wait_for(self.ws.recv(), timeout=0.5)
                        resp = json.loads(raw)

                        if resp.get("type") in ["response.text.delta", "transcription.delta"]:
                            file_transcript += resp.get("delta", "")
                        elif resp.get("type") in ["response.done", "transcription.done"]:
                            break
                except asyncio.TimeoutError:
                    pass

                logger.info(f"File processing complete. Transcript length: {len(file_transcript)}")
                return file_transcript if file_transcript else "No transcription received."

        except Exception as e:
            logger.error(f"File processing error: {e}")
            return f"Error processing file: {str(e)}"

client = VoxtralClient()

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("### 🎙️ Voxtral Real-time Translation Gateway")

    with gr.Tabs():
        with gr.Tab("🎤 Live Streaming"):
            gr.Markdown("Speak into your microphone for real-time translation")
            with gr.Row():
                audio_in = gr.Audio(sources=["microphone"], streaming=True, type="numpy")
                text_out = gr.Textbox(label="English Translation", lines=10)

            # Stream event with queue=False to prevent Stop hang in OpenShift
            audio_in.stream(
                fn=client.stream_audio,
                inputs=[audio_in],
                outputs=[text_out],
                show_progress="hidden",
                queue=False
            )

            audio_in.stop_recording(fn=lambda: client.transcript, outputs=[text_out])

        with gr.Tab("📁 File Upload"):
            gr.Markdown("Upload an audio file for translation")
            with gr.Row():
                with gr.Column():
                    file_in = gr.File(
                        label="Upload Audio File (WAV, MP3, FLAC, OGG, M4A, etc.)",
                        file_types=["audio"],
                        type="filepath"
                    )
                    process_btn = gr.Button("🚀 Translate File", variant="primary")
                with gr.Column():
                    file_out = gr.Textbox(label="English Translation", lines=15, interactive=False)

            process_btn.click(
                fn=client.process_file,
                inputs=[file_in],
                outputs=[file_out],
                queue=True
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)