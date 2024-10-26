# app.py
from googletrans import Translator
import torch
from transcription import transcribe_with_model, get_audio_length
from translation import translate_text
from srt_generator import generate_srt
from text_utils import format_text, split_text_into_paragraphs_by_sentence
from ui import interface

translator = Translator()

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

    general_transcription_result = transcribe_with_model(audio_file, model_choice)

    if generate_srt_checkbox:
        srt_content = generate_srt(general_transcription_result['segments'], target_language)
        return f"Audio Length: {audio_length} seconds\n\n" + format_text(srt_content)
    else:
        transcribed_text = ''.join([segment['text'] for segment in general_transcription_result['segments']])
        translated_text = translate_text(transcribed_text, target_language)
        return f"Audio Length: {audio_length} seconds\n\n" + format_text(translated_text)

if __name__ == "__main__":
    interface.launch()
