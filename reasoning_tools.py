import json
from collections import defaultdict

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, parse_obj_as
from tqdm import tqdm

from languages import language_codes
from translations import translate

system_message = "You are an expert in space exploration"


def get_documents():
    articles = [
        "Falcon 9",
        # "International Space Station",
        # "Space Shuttle",
    ]

    return WikipediaReader().load_data(
        pages=articles, auto_suggest=False
    )


def get_dataset(documents, num_questions_per_chunk=1):
    gpt_35_llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3)

    dataset_generator = RagDatasetGenerator.from_documents(
        documents,
        # question_gen_query=QUESTION_GEN_PROMPT,
        llm=gpt_35_llm,
        num_questions_per_chunk=num_questions_per_chunk,
        workers=1,
        show_progress=True,
    )
    return dataset_generator.generate_dataset_from_nodes()


class DatasetSample(BaseModel):
    question: str
    true_answer: str

    scores: dict[str, float] = {}
    answers: dict[str, str] = {}

    def set_lang(self, lang_code: str, answer: str, score: float):
        self.scores[lang_code] = score
        self.answers[lang_code] = answer

    def get_lang(self, lang_code: str):
        return self.scores[lang_code], self.answers[lang_code]


def load_dataset_samples(filename: str) -> list[DatasetSample]:
    with open(filename, 'r') as f:
        data = json.load(f)

    samples = parse_obj_as(list[DatasetSample], data)

    return samples


def save_dataset_samples(samples, filename: str):
    with open(filename, 'w') as f:
        json.dump([q.dict() for q in samples], f, indent=4)


async def evaluate_result(question, _answer, predicted_answer):
    gpt_4_llm = OpenAI(temperature=0, model="gpt-4")

    gpt4_judge = CorrectnessEvaluator(llm=gpt_4_llm)

    result = await gpt4_judge.aevaluate(
        query=question,
        reference=_answer,
        response=predicted_answer,
    )

    return result.score


def convert_dataset_to_samples(dataset):
    all_questions = []

    for ex in dataset.examples:
        sample = DatasetSample(question=ex.query, true_answer=ex.reference_answer)
        all_questions.append(sample)

    return all_questions

async def evaluate_samples(model_name: str, all_questions: list[DatasetSample]):
    llm = OpenAI(model=model_name, temperature=0)

    for sample in tqdm(all_questions):
        print(sample.question)
        for lang_code in language_codes:
            if lang_code in sample.scores:
                continue

            translated_question = translate(sample.question, lang_code)

            messages = [
                ChatMessage(
                    role="system", content=system_message
                ),
                ChatMessage(role="user", content=translated_question),
            ]

            answer = llm.chat(messages).message.content
            answer_eng = translate(answer, "EN-GB")

            score = await evaluate_result(sample.question, sample.true_answer, answer_eng)

            print(lang_code)
            print(score)
            print(answer_eng)

            sample.set_lang(lang_code, answer, score)


def compute_country_avg(all_questions: list[DatasetSample]):
    avg_countries = defaultdict(list)
    for sample in all_questions:
        for lang_code, score in sample.scores.items():
            avg_countries[lang_code].append(score)

    avg_countries_val = {}

    for lang_code, scores in avg_countries.items():
        avg_countries_val[lang_code] = sum(scores) / len(scores)

    return avg_countries_val