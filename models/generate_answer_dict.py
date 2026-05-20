"""
Generate mobile_answer_dict.json with all 10 languages.
Uses deep-translator (Google) to auto-translate English answers.

Usage: python generate_answer_dict.py
"""
import json, os, sys, time
from deep_translator import GoogleTranslator
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.vqa_answers import ANSWER_TEXTS_EN, ANSWER_TEXTS_VI, ANSWER_TEXTS_TH

INDIC_LANGS = ["hi", "te", "kn", "ta", "ml", "mr", "bn"]
OUTPUT = os.path.join(ROOT, "app", "static", "mobile_answer_dict.json")


def translate_all():
    result = {
        "en": {str(k): v for k, v in ANSWER_TEXTS_EN.items()},
        "vi": {str(k): v for k, v in ANSWER_TEXTS_VI.items()},
        "th": {str(k): v for k, v in ANSWER_TEXTS_TH.items()},
    }

    for lang in INDIC_LANGS:
        print(f"\n🌐 Translating to [{lang}]...")
        result[lang] = {}
        translator = GoogleTranslator(source="en", target=lang)
        for aid in tqdm(sorted(ANSWER_TEXTS_EN.keys()), desc=f"  {lang}"):
            try:
                result[lang][str(aid)] = translator.translate(ANSWER_TEXTS_EN[aid])
                time.sleep(0.2)
            except Exception as e:
                print(f"  ✗ [{aid}] failed: {e}")
                result[lang][str(aid)] = ANSWER_TEXTS_EN[aid]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {OUTPUT}")
    print(f"   Languages: {len(result)}, Entries/lang: {len(result['en'])}")


if __name__ == "__main__":
    translate_all()
