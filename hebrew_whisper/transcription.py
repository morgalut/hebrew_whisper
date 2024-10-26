# transcription.py
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
import shutil
from pydub import AudioSegment
import torch
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from text_utils import format_text, split_text_into_paragraphs_by_sentence
from srt_generator import generate_srt
from translation import translate_text

SAMPLING_RATE = 16000
general_model_name = 'large-v2'
hebrew_model_name = 'ivrit-ai/whisper-v2-d3-e3'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
general_model = whisper.load_model(general_model_name, device=device)
hebrew_processor = WhisperProcessor.from_pretrained(hebrew_model_name)
hebrew_model = WhisperForConditionalGeneration.from_pretrained(hebrew_model_name).to(device)

def transcribe_with_model(audio_file_path, model_choice):
    if model_choice == 'General Model':
        return transcribe_with_general_model(audio_file_path)
    else:
        return transcribe_with_hebrew_model(audio_file_path)

def transcribe_with_general_model(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    audio_numpy = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    audio_numpy = librosa.resample(audio_numpy, orig_sr=audio.frame_rate, target_sr=16000)

    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_file_name = tmpfile.name
            sf.write(temp_file_name, audio_numpy, 16000)

        transcription_result = general_model.transcribe(temp_file_name, language="he")
        return transcription_result
    finally:
        if temp_file_name:
            os.remove(temp_file_name)

def transcribe_with_hebrew_model(audio_file_path):
    audio, sr = librosa.load(audio_file_path, sr=SAMPLING_RATE)
    audio_numpy = np.array(audio)
    temp_dir = tempfile.mkdtemp()
    transcribed_segments = []

    for i in range(0, len(audio_numpy), SAMPLING_RATE * 30):
        chunk = audio_numpy[i:i + SAMPLING_RATE * 30]
        chunk_path = os.path.join(temp_dir, f"chunk_{i // (SAMPLING_RATE * 30)}.wav")
        sf.write(chunk_path, chunk, samplerate=SAMPLING_RATE)
        input_features = hebrew_processor(chunk, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features.to(device)
        predicted_ids = hebrew_model.generate(input_features, num_beams=5)
        chunk_text = hebrew_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcribed_segments.append({
            "start": i / SAMPLING_RATE,
            "end": min((i + SAMPLING_RATE * 30) / SAMPLING_RATE, len(audio_numpy) / SAMPLING_RATE),
            "text": chunk_text,
            "id": i // (SAMPLING_RATE * 30)
        })

    shutil.rmtree(temp_dir)
    return {"segments": transcribed_segments}

def get_audio_length(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000.0

def transcribe_and_translate(audio_file, target_language, model_choice, generate_srt_checkbox):
    if not target_language:
        return format_text("Please choose a Target Language")

    audio_length = get_audio_length(audio_file)

    if torch.cuda.is_available():
        print("GPU is available")
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU name: {gpu_info.name}")
        print(f"GPU memory usage before transcription:")
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Always transcribe with the general model to get accurate timestamps
    general_transcription_result = transcribe_with_model(audio_file, model_choice)

    if generate_srt_checkbox:
        srt_content = generate_srt(general_transcription_result['segments'], target_language)
        return f"Audio Length: {audio_length} seconds\n\n" + format_text(srt_content)
    else:
        transcribed_text = ''.join([segment['text'] for segment in general_transcription_result['segments']])
        translated_text = translate_text(transcribed_text, target_language)
        return f"Audio Length: {audio_length} seconds\n\n" + format_text(translated_text)
