from tokens import remove_special_characters
import numpy as np
from languages import num_languages, language_codes, language_names, non_latin_languages, language_codes_latin, \
    language_names_latin
from embedding_models import OpenAIEmbeddingModel, OpenAIEmbeddingModel3Small, all_embedding_models
import matplotlib.pyplot as plt
import seaborn as sns

DPI = 60


def find_min_max_from_dict_matrices(dict_matrices):
    overall_min = np.inf
    overall_max = -np.inf

    for key, matrix in dict_matrices.items():
        current_min = np.min(matrix)
        current_max = np.max(matrix)

        if current_min < overall_min:
            overall_min = current_min
        if current_max > overall_max:
            overall_max = current_max

    return overall_min, overall_max


def plot_confusion_matrices(matrices_dict, output_path: str, only_latin_langs=False):
    num_matrices = len(matrices_dict)

    num_rows = 3
    num_columns = -(-num_matrices // num_rows)

    min_value, max_value = find_min_max_from_dict_matrices(matrices_dict)
    plt.figure(figsize=(20 * num_columns, 20 * num_rows))

    if only_latin_langs:
        lang_names = language_names_latin
    else:
        lang_names = language_names

    lang_num = len(lang_names)

    for i, (plot_name, relationship_matrix) in enumerate(matrices_dict.items(), 1):
        plt.subplot(num_rows, num_columns, i)
        sns.heatmap(relationship_matrix, annot=True, fmt=".2f", cmap='viridis', vmin=min_value, vmax=max_value, cbar=False)
        plt.xticks(np.arange(lang_num) + 0.5, lang_names, rotation=90)
        plt.yticks(np.arange(lang_num) + 0.5, lang_names, rotation=0)
        plt.title(plot_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    # plt.show()


def compute_dict_array_diff(dict1, dict2):
    diff_dict = {}
    for key in dict1.keys():
        if key in dict2:
            # Compute the difference of the arrays and store it in the new dictionary
            diff_dict[key] = dict1[key] - dict2[key]
    return diff_dict
