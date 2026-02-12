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

    async def connect(self, send_commit=False, wait_for_session=True):
        ws_url = VLLM_URL.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/realtime"):
            ws_url = f"{ws_url.rstrip('/')}/realtime"

        # Increased timeout and ping for OpenShift stability
        self.ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=20)

        # Wait for session.created from server (only for initial connection)
        if wait_for_session:
            session_msg = await self.ws.recv()
            logger.info(f"Received: {session_msg[:100]}")

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

        # Only send commit for file uploads, not for streaming
        if send_commit:
            await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            logger.info("Connection established with Server VAD enabled (with commit).")
        else:
            logger.info("Connection established with Server VAD enabled.")

    def get_transcript(self):
        """Return the current transcript (for stop button)."""
        return self.transcript

    async def stream_audio(self, audio):
        if audio is None:
            return self.transcript

        async with self.lock:
            try:
                # Check connection state properly
                is_open = False
                if self.ws is not None:
                    # websockets library uses .open (bool) or .closed (bool) depending on version
                    if hasattr(self.ws, 'open'):
                        is_open = self.ws.open
                    elif hasattr(self.ws, 'closed'):
                        is_open = not self.ws.closed
                    else:
                        # Fallback: assume it's open if we have a ws object
                        is_open = True

                if not is_open:
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
                        if t in ["response.text.delta", "response.audio_transcription.delta", "transcription.delta"]:
                            delta = resp.get("delta", "") or resp.get("transcript", "")
                            if delta:
                                logger.info(f"Streaming: got delta '{delta[:50]}'")
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

            # Add silence at the end to help Server VAD detect end of speech
            silence = np.zeros(8000, dtype=np.int16)  # 0.5 seconds of silence
            audio_data = np.concatenate([audio_data, silence])

            # Create a NEW dedicated connection for file upload (not shared with streaming)
            ws_url = VLLM_URL.replace("https://", "wss://").replace("http://", "ws://")
            if not ws_url.endswith("/realtime"):
                ws_url = f"{ws_url.rstrip('/')}/realtime"

            file_ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=20)

            # Wait for session.created
            await file_ws.recv()

            # Send session update
            init_event = {
                "type": "session.update",
                "model": MODEL_NAME,
                "session": {
                    "modalities": ["text"],
                    "instructions": "Translate audio to English. Output only the translation.",
                    "input_audio_format": "pcm16",
                    "turn_detection": {"type": "server_vad"}
                }
            }
            await file_ws.send(json.dumps(init_event))

            # Send initial commit to signal readiness
            await file_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            logger.info("File upload: dedicated connection established")

            # Send audio in chunks with small delays to simulate real-time
            chunk_size = 16000  # 1 second at 16kHz
            file_transcript = ""

            logger.info(f"Sending {len(audio_data)} samples in chunks...")
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                audio_b64 = base64.b64encode(chunk.tobytes()).decode("utf-8")

                await file_ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))

                # Small delay to avoid overwhelming the VAD
                await asyncio.sleep(0.05)

            logger.info("All audio sent. Waiting for transcription...")

            # Wait for transcription (Server VAD will trigger when it detects speech)
            # Listen for up to 10 seconds for transcription deltas
            start_time = asyncio.get_event_loop().time()
            timeout = 10.0

            try:
                while asyncio.get_event_loop().time() - start_time < timeout:
                    raw = await asyncio.wait_for(file_ws.recv(), timeout=1.0)
                    resp = json.loads(raw)
                    t = resp.get("type")

                    logger.info(f"Received response type: {t}")

                    if t == "transcription.delta":
                        delta = resp.get("delta", "")
                        if delta:
                            logger.info(f"Got transcription delta: {delta[:50]}...")
                            file_transcript += delta
                    elif t == "error":
                        error_msg = resp.get("error", {})
                        logger.error(f"vLLM Error: {error_msg}")
                        await file_ws.close()
                        return f"Error: {error_msg.get('message', str(error_msg))}"
                    elif t == "transcription.done":
                        logger.info("Transcription complete.")
                        break
            except asyncio.TimeoutError:
                # Check if we got any transcript
                if not file_transcript:
                    logger.info("Timeout waiting for transcription. Retrying once...")
                    # Give it one more second
                    try:
                        raw = await asyncio.wait_for(file_ws.recv(), timeout=2.0)
                        resp = json.loads(raw)
                        if resp.get("type") == "transcription.delta":
                            file_transcript += resp.get("delta", "")
                    except asyncio.TimeoutError:
                        pass

            # Close the dedicated file upload connection
            await file_ws.close()

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

            # When recording stops, the stream function will be called with None
            # and will return the current transcript, so no explicit handler needed
            # audio_in.stop_recording(fn=client.get_transcript, outputs=[text_out])

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