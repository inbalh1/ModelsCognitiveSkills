import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import seaborn as sns

access_token = ""

MODEL_NANE = "lmsys/vicuna-7b-v1.3"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def generate_prompt(sentence, question):
    prompt = (
        f"Sentence: The capital of France is one of the most visited cities in the world.\n"
        f"Question: What is the capital of France?\n"
        f"Answer: Paris\n\n"
        f"Sentence: Water is composed of two hydrogen atoms and one oxygen atom.\n"
        f"Question: What is the chemical symbol for water?\n"
        f"Answer: H₂O\n\n"
        f"Sentence: The Earth is divided into large landmasses known as continents.\n"
        f"Question: How many continents are there on Earth?\n"
        f"Answer: Seven\n"
        f"Sentence: {sentence}\n"
        f"Question: {question}\n"
        f"Answer: The")

    return prompt


def predict_answer(model, tokenizer, sentence, correct_option, incorrect_option, question, num_tokens=5):
    prompt = generate_prompt(sentence, question)
    input_ids_sentence = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    correct_tokens = tokenizer(correct_option, return_tensors="pt").input_ids.squeeze()
    incorrect_tokens = tokenizer(incorrect_option, return_tensors="pt").input_ids.squeeze()

    # Get the possible token IDs (correct and incorrect tokens)
    correct_token_ids = correct_tokens.tolist()
    incorrect_token_ids = incorrect_tokens.tolist()
    valid_token_ids = list(set(correct_token_ids + incorrect_token_ids))
    if tokenizer.bos_token_id is not None:
        valid_token_ids.remove(tokenizer.bos_token_id)

    # Initialize the predicted tokens list with the sentence input
    generated_ids = input_ids_sentence
    answer_ids = tokenizer.encode("The")
    is_first_diff_token = False

    # here we are generting sequence in num_tokens length, to handle multiple tokens answers
    # in the accuracy calculation we will ceck only the first token of the answer
    for it in range(num_tokens):
        if len(correct_token_ids) > it and len(incorrect_token_ids) > it and correct_token_ids[it] == \
                incorrect_token_ids[it]:
            if correct_token_ids[it] != tokenizer.bos_token_id:
                generated_ids = torch.cat(
                    (generated_ids, torch.tensor([[correct_token_ids[it]]], device=generated_ids.device)), dim=-1)
                answer_ids.append(correct_token_ids[it])
            continue

        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits

        next_token_logits = logits[:, -1, :]
        next_token_logits_valid = next_token_logits[:, valid_token_ids]

        # Extract logits only for the tokens in correct and incorrect options and calc the probabilities
        if not is_first_diff_token:
            is_first_diff_token = True
            correct_first_token = correct_token_ids[it]
            incorrect_first_token = incorrect_token_ids[it]
            probabilities_token_ids = [correct_first_token, incorrect_first_token]
            probabilities_logits = next_token_logits[:, probabilities_token_ids]
            # calculate probabilities over first different tokens
            probabilities = torch.softmax(probabilities_logits, dim=-1)
            probabilities_list = probabilities.squeeze().tolist()
            token_probabilities = {token_id: prob for token_id, prob in
                                   zip(probabilities_token_ids, probabilities_list)}

        # Find the argmax of the valid token logits
        predicted_token_id = valid_token_ids[torch.argmax(next_token_logits_valid)]

        predicted_token_id_tensor = torch.tensor([[predicted_token_id]], device=generated_ids.device)
        answer_ids.append(predicted_token_id)
        generated_ids = torch.cat((generated_ids, predicted_token_id_tensor), dim=-1)

    predicted_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return predicted_text, token_probabilities, correct_first_token, incorrect_first_token


def predict_and_calculate_correct_predictions(model, tokenizer, data_df,
                                              output_path='output.csv', should_print=False):
    correct_predictions = 0
    total_predictions = 0

    output = []

    for index, row in data_df.iterrows():
        sentence = row['sentence']
        correct_option = row['correct_option']
        incorrect_option = row['incorrect_option']
        true_answer = row['answer']
        question = row['question']

        predicted_answer, token_probabilities, correct_token, incorrect_token = \
            predict_answer(model, tokenizer, sentence, correct_option, incorrect_option, question)

        is_correct_prediction = predicted_answer.startswith(true_answer)
        if is_correct_prediction:
            correct_predictions += 1

        total_predictions += 1

        output.append({
            # Add all columns from original df, and some additional information
            'sentence': row['sentence'],
            'question': row['question'],
            'set_id': row['set_id'],
            'type': row['type'],
            'answer': row['answer'],
            'correct_option': correct_option,
            'correct_first_token': correct_token,
            'correct_probability': token_probabilities.get(correct_token, 0),
            'incorrect_option': incorrect_option,
            'incorrect_first_token': incorrect_token,
            'incorrect_probability': token_probabilities.get(incorrect_token, 0),
            'predicted_answer': predicted_answer,
            'is_correct_prediction': is_correct_prediction
        })

    output_df = pd.DataFrame(output)
    output_df.to_csv(output_path, index=False)

    return correct_predictions, total_predictions


def calc_accuracy(correct_predictions, total_predictions, print=False):
    accuracy = correct_predictions / total_predictions * 100
    return accuracy


def plot_acc(acc_dict, title, xlabels, res_dir):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the barplot
    bars = ax.bar(list(acc_dict.keys()), list(acc_dict.values()), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                  edgecolor='black', alpha=0.85)

    # Add the accuracy values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Place the text at the top of the bar
                    xytext=(0, 3),  # Slightly above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

    # Set the title and labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabels, fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)

    # Improve layout and aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4c4c4c')
    ax.spines['bottom'].set_color('#4c4c4c')
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)

    plt.tight_layout()

    # Save the plot
    filename = title.lower().replace(' ', '_') + '.png'
    file_path = os.path.join(res_dir, filename)
    plt.savefig(file_path)

    # Show the plot
    plt.show()


def plot_grouped_barplot(categories, group_values, group_labels, xlabel, title, res_dir):
    """
    Function to create a grouped bar plot for categories with multiple groups of values.

    Parameters:
    - categories: A list of categories (e.g., different items, labels, etc.).
    - group_values: A list of lists, where each sublist contains values for a specific group.
                    Each sublist should correspond to a single group across all categories.
    - group_labels: A list of labels corresponding to each group.

    Example:
    plot_grouped_barplot(['Category 1', 'Category 2'], [[1.97, 0.61], [0.61, 0.51]], ['Group 1', 'Group 2'])
    """

    # Number of categories and number of groups
    n_categories = len(categories)
    n_groups = len(group_values)

    # Make sure the number of group values matches the number of labels
    assert n_groups == len(group_labels), "The number of group value sets must match the number of group labels"

    # Create bar width and position for grouped bars
    bar_width = 0.8 / n_groups  # Adjust bar width based on the number of groups
    index = np.arange(n_categories)

    # Plotting
    fig, ax = plt.subplots()

    # Loop through each group and plot its bars
    bars = []
    for i in range(n_groups):
        bars.append(ax.bar(index + i * bar_width, group_values[i], bar_width, label=group_labels[i]))

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(categories)

    # Increase y-axis limit for better spacing
    max_value = max([max(group) for group in group_values])
    ax.set_ylim(0, max_value * 1.4)  # Set y-limit a bit higher than the maximum value to allow space for labels

    # Add legend
    ax.legend()

    # Optionally add values above bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for bar_group in bars:
        add_labels(bar_group)

    # Display the plot
    plt.tight_layout()

    # Save the plot
    filename = title.lower().replace(' ', '_') + '.png'
    file_path = os.path.join(res_dir, filename)
    plt.savefig(file_path)

    # Show the plot after saving
    plt.show()


def plot_confidence(confidence, res_dir, title, color="blue"):
    plt.figure(figsize=(10, 6))

    # Plot histogram for first list
    sns.histplot(confidence, bins=20, kde=False, color=color, alpha=0.5, label='right', stat="density")

    # Add labels and title
    plt.title(title)
    plt.xlabel('Probability')
    plt.ylabel('Density')

    # Save the plot
    filename = title.lower().replace(' ', '_') + '.png'
    file_path = os.path.join(res_dir, filename)
    plt.savefig(file_path)
    # Show the plot
    plt.show()


def plot_confidence_grid(df, res_dir):
    # Define the types and grid size
    types = [
        "first_center_com", "second_center_com",
        "first_center_mid", "second_center_mid",
        "first_center_sim", "second_center_sim",
        "first_center_sim_rev", "second_center_sim_rev"
    ]

    # Create a grid of 4 rows and 4 columns (since we have 8 types, each with 2 plots)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.tight_layout(pad=5.0)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, t in enumerate(types):
        # For wrong answers
        wrong = df[(df["type"] == t) & (df["is_correct_prediction"] == False)]
        median_wrong = wrong["incorrect_probability"].median()  # Calculate median for wrong
        sns.histplot(wrong["incorrect_probability"], bins=20, binwidth=0.025, kde=False, color='blue', alpha=0.5,
                     label='incorrect', stat="percent", ax=axes[i * 2])
        axes[i * 2].axvline(median_wrong, color='red', linestyle='--', linewidth=2)  # Add vertical line at median
        axes[i * 2].text(median_wrong, 90, f'{median_wrong:.2f}', color='red', fontsize=10, va='center', ha='right',
                         backgroundcolor='white')  # Add median value text
        axes[i * 2].set_title(f'Confidence wrong {t}')
        axes[i * 2].set_xlabel('Probability')
        axes[i * 2].set_ylabel('Percent')
        axes[i * 2].set_xlim(0, 1)
        axes[i * 2].set_ylim(0, 100)

        # For correct answers
        right = df[(df["type"] == t) & (df["is_correct_prediction"] == True)]
        median_right = right["correct_probability"].median()  # Calculate median for correct
        sns.histplot(right["correct_probability"], bins=20, binwidth=0.025, kde=False, color='green', alpha=0.5,
                     label='correct', stat="percent", ax=axes[i * 2 + 1])
        axes[i * 2 + 1].axvline(median_right, color='red', linestyle='--', linewidth=2)  # Add vertical line at median
        axes[i * 2 + 1].text(median_right, 90, f'{median_right:.2f}', color='red', fontsize=10, va='center', ha='right',
                             backgroundcolor='white')  # Add median value text
        axes[i * 2 + 1].set_title(f'Confidence right {t}')
        axes[i * 2 + 1].set_xlabel('Probability')
        axes[i * 2 + 1].set_ylabel('Percent')
        axes[i * 2 + 1].set_xlim(0, 1)
        axes[i * 2 + 1].set_ylim(0, 100)

    # Save the entire grid plot
    filename = 'confidence_grid.png'
    file_path = os.path.join(res_dir, filename)
    plt.savefig(file_path)
    plt.show()


def run(data_path, out_path):
    df = pd.read_csv(data_path)
    correct_predictions_all, total_predictions_all = predict_and_calculate_correct_predictions(model, tokenizer, df,
                                                                                               output_path=out_path)
    # Run on only a part of the df
    # df_first = df[df["type"] == "first_center_sim"]
    # correct_predictions_first, total_predictions_first = predict_and_calculate_correct_predictions(model, tokenizer, df_first, output_path='first_center_output.csv')
    # print(correct_predictions_first)
    # print(total_predictions_first)
    return correct_predictions_all, total_predictions_all


def draw_all_plots(data_path, out_path, res_dir):
    # Run the model to get predictions
    correct_predictions_all, total_predictions_all = run(data_path, out_path)

    # Split according to wording and center types
    df = pd.read_csv(out_path)

    correct_preds_dict = {}
    total_preds_dict = {}

    for sen_type in df["type"].unique():
        sen_type_df = df[df["type"] == sen_type]
        correct_predictions = len(sen_type_df[sen_type_df["is_correct_prediction"] == 1])
        correct_preds_dict[sen_type] = correct_predictions
        total_predictions = len(sen_type_df)
        total_preds_dict[sen_type] = total_predictions

    # Sanity check
    assert (sum(correct_preds_dict.values()) == correct_predictions_all)
    assert (sum(total_preds_dict.values()) == total_predictions_all)

    # plots
    wordings = ["com", "mid", "sim", "sim_rev"]
    all_center_types = ["first_center", "second_center"]

    # Calculate accuracy for each wording separately
    wording_accuracy = {}

    for wording in wordings:
        correct_predictions = 0
        total_predictions = 0
        for center_type in all_center_types:
            correct_predictions += correct_preds_dict[center_type + "_" + wording]
            total_predictions += total_preds_dict[center_type + "_" + wording]
        wording_accuracy[wording] = calc_accuracy(correct_predictions, total_predictions)

    plot_acc(wording_accuracy, "Accuracy according to wording", "wordings", res_dir)

    # Compare according to different types
    center_types_accuracy = {}

    for center_type in all_center_types:
        correct_predictions = 0
        total_predictions = 0
        for wording in wordings:
            correct_predictions += correct_preds_dict[center_type + "_" + wording]
            total_predictions += total_preds_dict[center_type + "_" + wording]
        center_types_accuracy[center_type] = calc_accuracy(correct_predictions, total_predictions)

    plot_acc(center_types_accuracy, "Accuracy according to center type", "center type", res_dir)

    # Draw a plot of accuracy by wording category and center type as group

    acc_by_wording_center_type = []

    for center_type in all_center_types:
        wording_acc = []
        for wording in wordings:
            correct_predictions = correct_preds_dict[center_type + "_" + wording]
            total_predictions = total_preds_dict[center_type + "_" + wording]
            wording_acc.append(calc_accuracy(correct_predictions, total_predictions))
        acc_by_wording_center_type.append(wording_acc)

    plot_grouped_barplot(wordings, acc_by_wording_center_type, all_center_types, "wordings",
                         "Accuracy by wording and center type", res_dir)

    # Draw a plot of accuracy by center type as category and wording as group
    acc_by_center_type_wording = []

    for wording in wordings:
        center_type_acc = []
        for center_type in all_center_types:
            correct_predictions = correct_preds_dict[center_type + "_" + wording]
            total_predictions = total_preds_dict[center_type + "_" + wording]
            center_type_acc.append(calc_accuracy(correct_predictions, total_predictions))
        acc_by_center_type_wording.append(center_type_acc)

    plot_grouped_barplot(all_center_types, acc_by_center_type_wording, wordings, "center types",
                         "Accuracy by center type and wording", res_dir)

    # Check model confidence for right answers
    right = df[(df["is_correct_prediction"] == True)]
    plot_confidence(right["correct_probability"], res_dir, title='Model confidence for right answers')

    # Check model confidence for wrong answers
    wrong = df[(df["is_correct_prediction"] == False)]
    plot_confidence(wrong["incorrect_probability"], res_dir, title='Model confidence for wrong answers', color="green")

    plot_confidence_grid(df, res_dir)


def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    import torch

    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model name', required=False, nargs='?')
    parser.add_argument('-d', '--data_path', type=str, help='The model name', required=True, nargs='?')
    parser.add_argument('-n', '--num_gpus', type=str, help="Num of gpus to load the model on", default="1")
    parser.add_argument('-x', '--max_gpu_memory', type=float, default=None)

    args = parser.parse_args()

    models_path = args.model.split(',')
    nums_gpus = [int(i) for i in args.num_gpus.split(',')]

    assert len(models_path) == len(nums_gpus), "Some models have their num of GPUs unspecified"

    for i in range(len(models_path)):
        model_path = models_path[i]
        print(model_path)
        num_gpus = nums_gpus[i]

        tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {'torch_dtype': torch.float16}
        if num_gpus != 1:
            model_kwargs['device_map'] = 'sequential'
            available_gpu_memory = get_gpu_memory(num_gpus)
            model_kwargs["max_memory"] = {i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(num_gpus)}
        model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token, **model_kwargs)
        if num_gpus == 1:
            model.to(DEVICE)

    # Create the directory for results if it doesn't exist
        res_dir = os.path.join('results', model_path.replace("/", "_"))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    # data_df = pd.read_csv(args.data_path)
        out_path = os.path.join(res_dir, "output.csv")
        try:
            draw_all_plots(args.data_path, out_path, res_dir)
        except Exception as e:
            print(e)
            print(f"Failed for model {model_path}")



    


