from datasets import load_dataset

# Load the CLINC150 training split
dataset = load_dataset("clinc_oos", "small", split="train")

# Get the intent label names
intent_names = dataset.features['intent'].names  # list of 150 intent class names

# Show the first 10 examples
for i in range(300):
    text = dataset[i]['text']
    label_id = dataset[i]['intent']
    label_name = intent_names[label_id]

    print({
        "content": text,
        "labels": label_name
    })
