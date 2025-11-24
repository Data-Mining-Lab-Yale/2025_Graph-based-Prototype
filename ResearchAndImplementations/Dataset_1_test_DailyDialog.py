from datasets import load_dataset

# Load dataset (trust_remote_code required as of 2024+)
dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)

# Label mappings
intent_map = {
    0: "Inform",
    1: "Question",
    2: "Directive",
    3: "Commissive",
    4: "Greeting",
    5: "Closing"
}

emotion_map = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral"
}

# Show 3 conversations
for i in range(3):
    utterances = dataset[i]['dialog']
    act_labels = dataset[i]['act']
    emotion_labels = dataset[i]['emotion']

    print(f"\n=== Conversation {i+1} ===")
    for j, (utter, act, emo) in enumerate(zip(utterances, act_labels, emotion_labels)):
        print({
            "content": utter,
            "labels": {
                "intent": intent_map.get(act, f"Unknown({act})"),
                "emotion": emotion_map.get(emo, f"Unknown({emo})")
            }
        })
