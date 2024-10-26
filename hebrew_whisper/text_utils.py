# text_utils.py
import re

def is_hebrew(text):
    return bool(re.search(r'[\u0590-\u05FF]', text))

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def format_text(text):
    if is_hebrew(text):
        return f'<div style="text-align: right; direction: rtl;">{text}</div>'
    elif is_arabic(text):
        return f'<div style="text-align: left; direction: rtl;">{text}</div>'
    else:
        return f'<div style="text-align: left; direction: ltr;">{text}</div>'

def split_lines(text, max_line_length=40):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if len(' '.join(current_line + [word])) <= max_line_length:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def format_srt_entry(segment_id, start_time, end_time, lines):
    srt_entry = f"{segment_id}\n{start_time} --> {end_time}\n" + "\n".join(lines) + "\n\n"
    return srt_entry

def split_text_into_parts(text, num_parts):
    words = text.split()
    avg_words_per_part = len(words) // num_parts
    parts = []
    for i in range(num_parts):
        part = ' '.join(words[i * avg_words_per_part:(i + 1) * avg_words_per_part])
        parts.append(part)
    if len(parts) < num_parts:
        parts.append(' '.join(words[num_parts * avg_words_per_part:]))
    return parts

def split_text_into_paragraphs_by_sentence(text, max_sentences_per_paragraph=5):
    # Split the text into sentences using a regular expression
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    paragraphs = []
    current_paragraph = []

    for sentence in sentences:
        current_paragraph.append(sentence)
        if len(current_paragraph) >= max_sentences_per_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    return paragraphs
