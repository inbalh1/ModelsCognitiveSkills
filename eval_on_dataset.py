import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Initialize the predicted tokens list with the sentence input
    generated_ids = input_ids_sentence
    answer_ids = tokenizer.encode("The")

    # here we are generting sequence in num_tokens length, to handle multiple tokens answers
    # in the accuracy calculation we will ceck only the first token of the answer
    for _ in range(num_tokens):
        with torch.no_grad():
            outputs = model(generated_ids).logits
        
        next_token_logits = outputs[:, -1, :]

        # Extract logits only for the tokens in correct and incorrect options
        valid_token_ids = correct_token_ids + incorrect_token_ids
        next_token_logits_valid = next_token_logits[:, valid_token_ids]

        # Find the argmax of the valid token logits
        predicted_token_id = valid_token_ids[torch.argmax(next_token_logits_valid)]

        predicted_token_id_tensor = torch.tensor([[predicted_token_id]], device=generated_ids.device)
        answer_ids.append(predicted_token_id)
        generated_ids = torch.cat((generated_ids, predicted_token_id_tensor), dim=-1)

    predicted_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return predicted_text


def predict_and_calculate_correct_predictions(model, tokenizer, data_df, print=False):
    correct_predictions = 0
    total_predictions = 0

    for index, row in data_df.iterrows():
        sentence = row['sentence']
        correct_option = row['correct_option']
        incorrect_option = row['incorrect_option']
        true_answer = row['answer']
        question = row['question']

        predicted_answer = predict_answer(model, tokenizer, sentence, correct_option, incorrect_option, question)
        if print:
          print(f"Sentence: {sentence}")
          print(f"Correct Option: {correct_option}")
          print(f"Incorrect Option: {incorrect_option}")
          print(f"True Answer: {true_answer}")
          print(f"Predicted Answer: {predicted_answer}")

        if predicted_answer.startswith(true_answer):
            correct_predictions += 1
            if print:
              print("Correct prediction!")
        else:
            if print:
              print("Incorrect prediction.")

        if print:
          print("-" * 50)

        total_predictions += 1

    return correct_predictions, total_predictions

def calc_accuracy(correct_predictions, total_predictions, print=False):
    accuracy = correct_predictions / total_predictions * 100
    if print:
      print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def plot_acc(names, values, title, xlabels, res_dir):
  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(8, 6))

  # Create the barplot
  bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black', alpha=0.85)

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

def draw_all_plots(df, res_dir):
    # Split according to wording and center types
    # com wording
    df_first = df[df["type"] == "first_center"]
    correct_predictions_first, total_predictions_first = predict_and_calculate_correct_predictions(model, tokenizer, df_first)

    df_second = df[df["type"] == "second_center"]
    correct_predictions_second, total_predictions_second = predict_and_calculate_correct_predictions(model, tokenizer, df_second)

    # nd wording
    df_first_nd = df[df["type"] == "first_center_nd"]
    correct_predictions_first_nd, total_predictions_first_nd = predict_and_calculate_correct_predictions(model, tokenizer, df_first_nd)

    df_second_nd = df[df["type"] == "second_center_nd"]
    correct_predictions_second_nd, total_predictions_second_nd = predict_and_calculate_correct_predictions(model, tokenizer, df_second_nd)

    # sim wording
    df_first_sim = df[df["type"] == "first_center_sim"]
    correct_predictions_first_sim, total_predictions_first_sim = predict_and_calculate_correct_predictions(model, tokenizer, df_first_sim)

    df_second_sim = df[df["type"] == "second_center_sim"]
    correct_predictions_second_sim, total_predictions_second_sim = predict_and_calculate_correct_predictions(model, tokenizer, df_second_sim)    
    
    # Compare accuracy between different wordings
    acc = calc_accuracy(correct_predictions_first + correct_predictions_second, total_predictions_first + total_predictions_second)
    acc_nd = calc_accuracy(correct_predictions_first_nd + correct_predictions_second_nd, total_predictions_first_nd + total_predictions_second_nd)
    acc_sim = calc_accuracy(correct_predictions_first_sim + correct_predictions_second_sim, total_predictions_first_sim + total_predictions_second_sim)

    plot_acc(["com", "nd", "sim"], [acc, acc_nd, acc_sim], "Accuracy according to wording", "wordings", res_dir)
    
        # Compare according to different types
    acc_first = calc_accuracy(correct_predictions_first + correct_predictions_first_nd + correct_predictions_first_sim,
                              total_predictions_first + total_predictions_first_nd + total_predictions_first_sim)

    acc_second = calc_accuracy(correct_predictions_second + correct_predictions_second_nd + correct_predictions_second_sim,
                              total_predictions_second + total_predictions_second_nd + total_predictions_second_sim)

    plot_acc(["first center", "second center"], [acc_first, acc_second], "Accuracy according to center type", "center type", res_dir)

    # Draw a plot of accuracy by wording category and center type as group
    wordings = ['com', 'nd', 'sim']
    types = ["first center", "second center"]
    accuracy_type_1 = [calc_accuracy(correct_predictions_first, total_predictions_first),
                       calc_accuracy(correct_predictions_first_nd, total_predictions_first_nd),
                       calc_accuracy(correct_predictions_first_sim, total_predictions_first_sim)]

    accuracy_type_2 = [calc_accuracy(correct_predictions_second, total_predictions_second),
                       calc_accuracy(correct_predictions_second_nd, total_predictions_second_nd),
                       calc_accuracy(correct_predictions_second_sim, total_predictions_second_sim)]

    values = [accuracy_type_1, accuracy_type_2]
    plot_grouped_barplot(wordings, values, types, "wordings", "Accuracy by wording and center type", res_dir)

    # Draw a plot of accuracy by center type as category and wording as group
    wordings = ['com', 'nd', 'sim']
    center_types = ["first center", "second center"]

    accuracy_com = [calc_accuracy(correct_predictions_first, total_predictions_first), #type 1
                       calc_accuracy(correct_predictions_second, total_predictions_second)] #type 2

    accuracy_nd = [calc_accuracy(correct_predictions_first_nd, total_predictions_first_nd), #type 1
                       calc_accuracy(correct_predictions_second_nd, total_predictions_second_nd)] #type 2

    accuracy_sim = [calc_accuracy(correct_predictions_first_sim, total_predictions_first_sim), #type 1
                       calc_accuracy(correct_predictions_second_sim, total_predictions_second_sim)] #type 2


    values = [accuracy_com, accuracy_nd, accuracy_sim]
    plot_grouped_barplot(types, values, wordings, "center types", "Accuracy by center type and wording", res_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model name', required=False, nargs='?')
    parser.add_argument('-d', '--data_path', type=str, help='The model name', required=True, nargs='?')

    args = parser.parse_args()

    model_path = args.model or MODEL_NANE

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(DEVICE)

    # Create the directory for results if it doesn't exist
    res_dir = model_path.replace("/", "_")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        

    data_df= pd.read_csv(args.data_path)
    # data_df = pd.from_csv(args.data_path)
    draw_all_plots(data_df, res_dir)



    


