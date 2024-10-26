# translation.py
from deep_translator import GoogleTranslator

def translate_text(text, target_lang):
    translations = {
        'Hebrew': 'he', 'English': 'en', 'Spanish': 'es', 'French': 'fr',
        'German': 'de', 'Portuguese': 'pt', 'Arabic': 'ar'
    }
    translated_text = GoogleTranslator(source='auto', target=translations[target_lang]).translate(text)
    return translated_text
