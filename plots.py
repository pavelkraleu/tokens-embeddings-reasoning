from embeddings_tools import find_min_max_from_dict_matrices
from languages import languages

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


DPI = 60


def plot_model_lang_distributions(lang_dict: dict, png_plot_path: str, plot_y_label: str, num_rows=3, palette='plasma'):
    num_models = len(lang_dict)
    num_columns = -(-num_models // num_rows)
    font_size = 20

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20 * num_columns, 15 * num_rows), constrained_layout=True, sharey=True)

    axes = axes.flatten() if num_models > 1 else [axes]

    for idx, (model_name, lang_data) in enumerate(lang_dict.items()):
        categories = [languages[code] for code in lang_data.keys()]
        values = list(lang_data.values())

        sns.barplot(x=categories, y=values, ax=axes[idx], palette=palette)
        axes[idx].set_title(model_name, fontsize=font_size*2)
        # axes[idx].set_xlabel('Languages', fontsize=font_size)
        axes[idx].set_ylabel(plot_y_label if idx % num_columns == 0 else '', fontsize=font_size*2)
        axes[idx].tick_params(axis='x', rotation=90, labelsize=font_size)
        axes[idx].tick_params(axis='y', labelsize=font_size)

    # Save the figure
    plt.savefig(png_plot_path, dpi=DPI)


def plot_as_separate_charts(lang_dict: dict, png_plot_path: str, plot_y_label: str):
    for i, [model_name, model_values] in enumerate(lang_dict.items()):
        new_dict = {model_name: model_values}

        plot_model_lang_distributions(
            new_dict,
            png_plot_path + f"_{i}.png",
            plot_y_label,
            num_rows=1
        )


def plot_boxplot_for_matrices(matrices_dict: dict, png_plot_path: str):
    num_matrices = len(matrices_dict)

    # Calculate the number of rows needed to plot all matrices
    num_rows = 3
    num_columns = -(-num_matrices // num_rows)

    min_value, max_value = find_min_max_from_dict_matrices(matrices_dict)

    # Setup the figure size based on the number of columns
    plt.figure(figsize=(20 * num_columns, 20 * num_rows))

    # Loop through the dictionary and create a box plot for the values in each matrix
    for i, (plot_name, relationship_matrix) in enumerate(matrices_dict.items(), 1):
        ax = plt.subplot(num_rows, num_columns, i)
        # Flatten the matrix to get all values in a single array
        matrix_values = relationship_matrix.flatten()
        # Create a boxplot of the flattened matrix values
        sns.boxplot(data=matrix_values)
        plt.title(plot_name)
        ax.set_ylim(min_value, max_value)

    plt.tight_layout()
    plt.savefig(png_plot_path, dpi=DPI)
    plt.show()
