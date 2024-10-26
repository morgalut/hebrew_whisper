# ui.py
import gradio as gr
from transcription import transcribe_and_translate

title = "Unlimited Length Transcription and Translation"
description = "With: whisper-large-v2 or ivrit-ai/whisper-v2-d3-e3 models | GUI by Mor Galut"

interface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(choices=['Hebrew', 'English', 'Spanish', 'French', 'German', 'Portuguese', 'Arabic'], 
                    label="Target Language", value='Hebrew'),
        gr.Dropdown(choices=['General Model', 'Hebrew Model'], 
                    label="Model Choice", value='Hebrew Model'),
        gr.Checkbox(label="Generate Hebrew SRT File")
    ],
    outputs=gr.HTML(label="Transcription / Translation / SRT Result"),
    title=title,
    description=description
)

interface.css = """
    #output_text, #output_text * {
        text-align: right !important;
        direction: rtl !important;
    }
"""
