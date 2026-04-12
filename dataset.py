# Why: load_dataset downloads and loads the IMDB dataset in a structured format.
from datasets import load_dataset

# Why: train_test_split helps us create reproducible member and non-member subsets.
from sklearn.model_selection import train_test_split


# Why: this function centralizes dataset loading and splitting so the rest of the project can reuse it.
def load_data(member_size=500, non_member_size=500, random_state=42):
    # Why: this loads the full IMDB dataset, including official train and test splits.
    dataset = load_dataset("imdb")

    # Why: these are the full text and label lists from the official training split.
    train_texts_full = dataset["train"]["text"]
    train_labels_full = dataset["train"]["label"]

    # Why: these are the full text and label lists from the official test split.
    test_texts_full = dataset["test"]["text"]
    test_labels_full = dataset["test"]["label"]

    # Why: members must come from the training split because the target model will see them during training.
    member_texts, _, member_labels, _ = train_test_split(
        train_texts_full,
        train_labels_full,
        train_size=member_size,
        stratify=train_labels_full,
        random_state=random_state,
    )

    # Why: non-members must come from the test split so they remain unseen by the target model.
    non_member_texts, _, non_member_labels, _ = train_test_split(
        test_texts_full,
        test_labels_full,
        train_size=non_member_size,
        stratify=test_labels_full,
        random_state=random_state,
    )

    # Why: returning these four objects gives downstream files exactly what they need.
    return member_texts, member_labels, non_member_texts, non_member_labels


# Why: this block lets you test dataset loading directly without affecting imports in other files.
if __name__ == "__main__":
    # Why: load the default controlled experiment split for a sanity check.
    member_texts, member_labels, non_member_texts, non_member_labels = load_data()

    # Why: these prints confirm the split sizes are correct.
    print("Members:", len(member_texts))
    print("Non-members:", len(non_member_texts))

    # Why: these prints help verify that the data looks valid.
    print("First member label:", member_labels[0])
    print("First member review preview:", member_texts[0][:200])