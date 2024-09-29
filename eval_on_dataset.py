import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def predict_and_calculate_accuracy(model, tokenizer, data_df):
    correct_predictions = 0
    total_predictions = 0

    for index, row in data_df.iterrows():
        sentence = row['sentence']
        correct_option = row['correct_option']
        incorrect_option = row['incorrect_option']
        true_answer = row['answer']
        question = row['question']

        predicted_answer = predict_answer(model, tokenizer, sentence, correct_option, incorrect_option, question)
        print(f"Sentence: {sentence}")
        print(f"Correct Option: {correct_option}")
        print(f"Incorrect Option: {incorrect_option}")
        print(f"True Answer: {true_answer}")
        print(f"Predicted Answer: {predicted_answer}")

        if predicted_answer.startswith(true_answer):
            correct_predictions += 1
            print("Correct prediction!")
        else:
            print("Incorrect prediction.")

        print("-" * 50)

        total_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions * 100
    return accuracy


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

    data_df = pd.from_csv(args.data_path)

    accuracy = predict_and_calculate_accuracy(model, tokenizer, data_df)
    print(f"Accuracy: {accuracy:.2f}%")

    


