# Why: this library lets you directly download datasets from Hugging Face
from datasets import load_dataset

# Why: train_test_split helps us create a smaller, controlled experiment.
from sklearn.model_selection import train_test_split

# Why: this line downloads and loads the IMDB dataset into memory
dataset = load_dataset("imdb")

# Why: confirms dataset structure and splits
print(dataset)

# Why: shows one sample so you understand the format
print(dataset["train"][0])

# Why: we separate texts and labels because the model needs inputs and targets independently.
train_texts_full = dataset["train"]["text"]
train_labels_full = dataset["train"]["label"]

test_texts_full = dataset["test"]["text"]
test_labels_full = dataset["test"]["label"]

# Why: these are the member samples the model will train on.
member_texts, _, member_labels, _ = train_test_split(
    train_texts_full,
    train_labels_full,
    train_size=500,
    stratify=train_labels_full,
    random_state=42
)

# Why: these are the non-member samples the model will never see during training.
non_member_texts, _, non_member_labels, _ = train_test_split(
    test_texts_full,
    test_labels_full,
    train_size=500,
    stratify=test_labels_full,
    random_state=42
)

# Why: these prints verify the sizes of the two groups.
print("Members:", len(member_texts))
print("Non-members:", len(non_member_texts))