import json
import re
import random
import nltk
from nltk.corpus import wordnet

# Load the dataset
def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Data Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_data(data):
    for intent in data["intents"]:
        intent["patterns"] = [clean_text(p) for p in intent["patterns"]]
        intent["responses"] = [clean_text(r) for r in intent["responses"]]
    return data

# Data Augmentation
def augment_text(text):
    words = text.split()
    augmented_text = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            augmented_text.append(synonym if synonym != word else word)
        else:
            augmented_text.append(word)
    return " ".join(augmented_text)

def augment_data(data):
    for intent in data["intents"]:
        new_patterns = [augment_text(p) for p in intent["patterns"]]
        intent["patterns"].extend(new_patterns)
    return data

# Splitting Dataset
def split_data(data, train_ratio=0.7, val_ratio=0.2):
    random.shuffle(data["intents"])
    train_size = int(train_ratio * len(data["intents"]))
    val_size = int(val_ratio * len(data["intents"]))
    return {
        "train": {"intents": data["intents"][:train_size]},
        "val": {"intents": data["intents"][train_size:train_size + val_size]},
        "test": {"intents": data["intents"][train_size + val_size:]}
    }

# Main function
def main():
    nltk.download("wordnet")
    data = load_data("intents.json")
    data = clean_data(data)
    data = augment_data(data)
    splits = split_data(data)

    # Save the outputs
    with open("intents_cleaned.json", "w") as file:
        json.dump(data, file, indent=4)
    with open("train.json", "w") as file:
        json.dump(splits["train"], file, indent=4)
    with open("val.json", "w") as file:
        json.dump(splits["val"], file, indent=4)
    with open("test.json", "w") as file:
        json.dump(splits["test"], file, indent=4)

    print("Preprocessing completed!")

if __name__ == "__main__":
    main()
