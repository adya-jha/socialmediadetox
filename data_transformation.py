import json

def transform_data(input_file, output_file):
    # Load the dataset
    with open(input_file, "r") as file:
        data = json.load(file)

    # Convert intents to input-output pairs
    formatted_data = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            for response in intent["responses"]:
                formatted_data.append({"input": f"User: {pattern}", "output": f"AI: {response}"})

    # Save formatted data
    with open(output_file, "w") as file:
        json.dump(formatted_data, file, indent=4)

    print(f"Formatted data saved to {output_file}!")

# Main function
if __name__ == "__main__":
    transform_data("train.json", "formatted_train.json")
    transform_data("val.json", "formatted_val.json")
    transform_data("test.json", "formatted_test.json")
