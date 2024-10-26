# srt_generator.py
import os
import datetime
from translation import translate_text
from text_utils import split_lines, format_srt_entry

def generate_srt(segments, target_language):
    srt_content = ""
    segment_id = 1
    max_line_length = 40

    for segment in segments:
        start_time_seconds = segment['start']
        end_time_seconds = segment['end']
        text = segment['text']

        lines = split_lines(text, max_line_length=max_line_length)
        while lines:
            current_lines = lines[:2]
            lines = lines[2:]

            start_time = str(datetime.timedelta(seconds=start_time_seconds)).split(".")[0] + ',000'
            end_time = str(datetime.timedelta(seconds=end_time_seconds)).split(".")[0] + ',000'
            translated_lines = [translate_text(line, target_language) for line in current_lines]

            srt_entry = format_srt_entry(segment_id, start_time, end_time, translated_lines)
            srt_content += srt_entry
            segment_id += 1

    os.makedirs("output", exist_ok=True)
    srt_file_path = os.path.join("output", "output.srt")
    with open(srt_file_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    srt_html_content = ""
    for line in srt_content.split('\n'):
        srt_html_content += f"<div>{line}</div>"

    return srt_html_content
