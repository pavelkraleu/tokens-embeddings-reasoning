import hashlib
import unicodedata
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer
from vertexai.generative_models import GenerativeModel

from languages import non_latin_languages


def remove_special_characters(input_str: str) -> str:
    normalized_str = unicodedata.normalize('NFKD', input_str)
    ascii_str = normalized_str.encode('ascii', 'ignore')
    return ascii_str.decode('ascii')


def calculate_percentage_differences(dict1: dict[dict], dict2: dict[dict]):
    percentage_differences = defaultdict(dict)

    for model in dict1.keys():
        for key in dict1[model]:
            lang_is_also_no_special_char = key not in dict2[model] or key in non_latin_languages
            if lang_is_also_no_special_char:
                continue

            difference = ((dict2[model][key] - dict1[model][key]) / dict1[model][key]) * 100
            percentage_differences[model][key] = difference
    return percentage_differences


class BaseTokenCounter(ABC):
    NAME = "TokenCounter"
    VERBOSE_NAME = "TokenCounter"
    OUTPUT_FRACTION_INPUT = 0.5

    def get_cached_token_count(self, text) -> int | None:
        md5_hash = hashlib.md5(text.encode()).hexdigest()

        destination_path = Path("token_counts") / f"{md5_hash}_{self.NAME.replace('/','-')}.txt"

        if not destination_path.exists():
            token_count = self.count_tokens(text)
            destination_path.write_text(str(token_count))

        return int(destination_path.read_text())

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def processing_cost_usd(self, text: str) -> float:
        pass


class GPTFamilyTokenCounter(BaseTokenCounter):
    input_cost_per_token_usd = None
    output_cost_per_token_usd = None

    def count_tokens(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.NAME)
        tokens = encoding.encode(text)
        return len(tokens)

    def processing_cost_usd(self, text: str) -> float:
        num_tokens = self.get_cached_token_count(text)

        input_tokens = num_tokens
        output_tokens = num_tokens * self.OUTPUT_FRACTION_INPUT

        return (input_tokens * self.input_cost_per_token_usd) + (output_tokens * self.output_cost_per_token_usd)


class HF7BTokenCounter(BaseTokenCounter):
    input_cost_per_token_usd = None
    output_cost_per_token_usd = None

    def count_tokens(self, text: str) -> int:
        return len(self.get_tokens(text))

    def get_tokens(self, text: str) -> int:
        tokenizer = Tokenizer.from_pretrained(self.NAME)
        return tokenizer.encode(text).ids

    def processing_cost_usd(self, text: str) -> float:
        num_tokens = self.get_cached_token_count(text)

        input_tokens = num_tokens
        output_tokens = num_tokens * self.OUTPUT_FRACTION_INPUT

        return (input_tokens * self.input_cost_per_token_usd) + (output_tokens * self.output_cost_per_token_usd)


class Mistral7BTokenCounter(HF7BTokenCounter):
    NAME = "mistralai/Mistral-7B-v0.1"
    VERBOSE_NAME = "Mistral-7B"

    input_cost_per_token_usd = 0.05 / 1_000_000
    output_cost_per_token_usd = 0.25 / 1_000_000


class Mixtral8x7BTokenCounter(HF7BTokenCounter):
    NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    VERBOSE_NAME = "Mixtral-8x7B"

    input_cost_per_token_usd = 0.30 / 1_000_000
    output_cost_per_token_usd = 1.00 / 1_000_000


class Llama7BTokenCounter(HF7BTokenCounter):
    NAME = "TheBloke/Llama-2-7B-Chat-GPTQ"
    VERBOSE_NAME = "Llama-2-7B"

    input_cost_per_token_usd = 0.05 / 1_000_000
    output_cost_per_token_usd = 0.25 / 1_000_000


class Llama70BTokenCounter(HF7BTokenCounter):
    NAME = "TheBloke/Llama-2-70B-Chat-GPTQ"
    VERBOSE_NAME = "Llama-2-70B"

    input_cost_per_token_usd = 0.65 / 1_000_000
    output_cost_per_token_usd = 2.75 / 1_000_000


class GPT4TokenCounter(GPTFamilyTokenCounter):
    NAME = "gpt-4"
    VERBOSE_NAME = "GPT-4"

    input_cost_per_token_usd = 0.03 / 1000
    output_cost_per_token_usd = 0.06 / 1000


class GPT4TurboTokenCounter(GPTFamilyTokenCounter):
    NAME = "gpt-4-0125-preview"
    VERBOSE_NAME = "GPT-4 Turbo"

    input_cost_per_token_usd = 0.01 / 1000
    output_cost_per_token_usd = 0.03 / 1000



class GPT35TurboTokenCounter(GPTFamilyTokenCounter):
    NAME = "gpt-3.5-turbo"
    VERBOSE_NAME = "GPT-3.5"

    input_cost_per_token_usd = 0.0005 / 1000
    output_cost_per_token_usd = 0.0015 / 1000


class GeminiPro10TokenCounter(BaseTokenCounter):
    NAME = "gemini-1.0-pro"
    VERBOSE_NAME = "Gemini Pro 1.0"

    input_cost_per_character_usd = 0.000125 / 1000
    output_cost_per_character_usd = 0.000375 / 1000

    def get_cached_billable_chars(self, text) -> int | None:
        md5_hash = hashlib.md5(text.encode()).hexdigest()

        destination_path = Path("billable_chars") / f"{md5_hash}_{self.NAME}.txt"

        if not destination_path.exists():
            token_count = self.billable_characters(text)
            destination_path.write_text(str(token_count))

        return int(destination_path.read_text())

    def count_tokens(self, text: str) -> int:
        model = GenerativeModel(self.NAME)
        return model.count_tokens(text).total_tokens

    def billable_characters(self, text: str) -> int:
        model = GenerativeModel(self.NAME)
        return model.count_tokens(text).total_billable_characters

    def processing_cost_usd(self, text: str) -> float:
        num_characters = self.get_cached_billable_chars(text)

        input_tokens = num_characters
        output_tokens = num_characters * self.OUTPUT_FRACTION_INPUT

        return (input_tokens * self.input_cost_per_character_usd) + (output_tokens * self.output_cost_per_character_usd)




all_tokenizers = [
    GPT4TokenCounter(),
    GPT4TurboTokenCounter(),
    GPT35TurboTokenCounter(),
    GeminiPro10TokenCounter(),
    Mistral7BTokenCounter(),
    Llama7BTokenCounter(),
    Mixtral8x7BTokenCounter(),
    Llama70BTokenCounter()
]

