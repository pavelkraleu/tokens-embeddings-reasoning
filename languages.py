non_latin_languages = ["UK", "EL", "BG", "ZH", "KO", "JA", "AR"]

languages = {
    "EN-GB": "English UK",
    "FR": "French",
    "ES": "Spanish",
    "PT-PT": "Portuguese",
    "DE": "German",
    "IT": "Italian",
    "NL": "Dutch",
    "DA": "Danish",
    "SV": "Swedish",
    "NB": "Norwegian",
    "FI": "Finnish",
    "ET": "Estonian",
    "LV": "Latvian",
    "LT": "Lithuanian",
    "PL": "Polish",
    "SK": "Slovak",
    "SL": "Slovenian",
    "CS": "Czech",
    "HU": "Hungarian",
    "RO": "Romanian",
    "BG": "Bulgarian",
    "EL": "Greek",
    "TR": "Turkish",
    "AR": "Arabic",
    "UK": "Ukrainian",
    "ZH": "Chinese",
    "KO": "Korean",
    "JA": "Japanese",
    "ID": "Indonesian",
}


languages_latin = {code: name for code, name in languages.items() if code not in non_latin_languages}

language_codes = list(languages.keys())
language_names = list(languages.values())

language_codes_latin = list(languages_latin.keys())
language_names_latin = list(languages_latin.values())

num_languages = len(language_codes)
