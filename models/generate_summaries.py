import json
import os
import sys
import time
from deep_translator import GoogleTranslator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

disease_base = {
    "Brown_Stem_Spot": "Brown Stem Spot detected. It is caused by the Bipolaris fungus.",
    "Anthracnose": "Anthracnose detected. Reddish-brown spots are visible on the stem.",
    "Soft_Rot": "Soft Rot detected. The tissue appears yellow, soft, and watery.",
    "Stem_Canker": "Stem Canker detected. Sunken lesions are present on the branches.",
    "Gray_Blight": "Gray Blight detected. Ash-gray spots with dark centers are visible.",
    "Healthy": "The plant appears healthy. No visible signs of disease."
}

INDIC_LANGS = ["hi", "te", "kn", "ta", "ml", "mr", "bn", "vi", "th"]

DISEASE_SUMMARIES = {}

for disease, text in disease_base.items():
    DISEASE_SUMMARIES[disease] = {"en": text}
    for lang in INDIC_LANGS:
        try:
            translator = GoogleTranslator(source="en", target=lang)
            translated = translator.translate(text)
            DISEASE_SUMMARIES[disease][lang] = translated
            time.sleep(0.2)
        except Exception as e:
            print(f"Failed {disease} to {lang}: {e}")
            DISEASE_SUMMARIES[disease][lang] = text

output_path = os.path.join(ROOT, "chatbot", "disease_summaries.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(DISEASE_SUMMARIES, f, ensure_ascii=False, indent=4)
print(f"Saved {output_path}")
