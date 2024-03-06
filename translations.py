import hashlib
import os
from pathlib import Path

import deepl


def translate(text: str, target_lang_code: str) -> str:
    md5_hash = hashlib.md5(text.encode()).hexdigest()

    destination_path = Path("translations") / f"{md5_hash}_{target_lang_code}.txt"

    if not destination_path.exists():
        translator = deepl.Translator(os.environ["DEEPL_KEY"])
        result = translator.translate_text(text, target_lang=target_lang_code)
        destination_path.write_text(result.text)

    return destination_path.read_text()
