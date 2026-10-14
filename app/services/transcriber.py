"""Khmer speech-to-text using a fine-tuned Whisper model."""
from __future__ import annotations

import io
import logging
import threading
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 10  # model was trained on clips of at most ~10s


class KhmerTranscriber:
    """Lazily loads the model on first use and transcribes audio bytes."""

    def __init__(self, model_id: str = "chitra168/whisper-small-khmer-mini"):
        self.model_id = model_id
        self._processor: WhisperProcessor | None = None
        self._model: WhisperForConditionalGeneration | None = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                logger.info("Loading Whisper model %s", self.model_id)
                self._processor = WhisperProcessor.from_pretrained(self.model_id)
                model = WhisperForConditionalGeneration.from_pretrained(self.model_id).eval()
                # The repo stores eos_token_id as a list, which breaks the
                # timestamp logits processor (it slices with it)
                eos = model.generation_config.eos_token_id
                if isinstance(eos, list):
                    model.generation_config.eos_token_id = eos[0]
                self._model = model

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        """Decode OGG/Opus (Telegram voice), MP3, WAV, ... into 16 kHz mono float32."""
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            g = gcd(sr, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
        return audio

    def transcribe(self, data: bytes) -> str:
        """Blocking — call via asyncio.to_thread."""
        self._load()
        audio = self._decode(data)
        step = SAMPLE_RATE * CHUNK_SECONDS
        chunks = [audio[i:i + step] for i in range(0, len(audio), step)]
        # Drop a tiny trailing fragment (<0.3s) — it only produces hallucinations
        if len(chunks) > 1 and len(chunks[-1]) < SAMPLE_RATE * 0.3:
            chunks.pop()

        texts = []
        for chunk in chunks:
            features = self._processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            with torch.inference_mode():
                ids = self._model.generate(
                    features, language="khmer", task="transcribe",
                    return_timestamps=True, max_new_tokens=440,
                )
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if text:
                texts.append(text)
        return " ".join(texts)
