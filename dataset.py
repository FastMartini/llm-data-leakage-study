# Why: load_dataset downloads and loads the IMDB dataset from Hugging Face.
from datasets import load_dataset

# Why: train_test_split lets us sample smaller, reproducible member and non-member subsets.
from sklearn.model_selection import train_test_split


# Why: putting the dataset logic in a function makes it reusable from train.py and attack.py.
def load_data(member_size=500, non_member_size=500, random_state=42):
    # Why: this downloads and loads the full IMDB dataset into memory.
    dataset = load_dataset("imdb")

    # Why: these are the full official train texts and labels that we will sample members from.
    train_texts_full = dataset["train"]["text"]
    train_labels_full = dataset["train"]["label"]

    # Why: these are the full official test texts and labels that we will sample non-members from.
    test_texts_full = dataset["test"]["text"]
    test_labels_full = dataset["test"]["label"]

    # Why: members are sampled from the training split because the target model will train on them.
    member_texts, _, member_labels, _ = train_test_split(
        train_texts_full,
        train_labels_full,
        train_size=member_size,
        stratify=train_labels_full,
        random_state=random_state,
    )

    # Why: non-members are sampled from the test split so they remain unseen during training.
    non_member_texts, _, non_member_labels, _ = train_test_split(
        test_texts_full,
        test_labels_full,
        train_size=non_member_size,
        stratify=test_labels_full,
        random_state=random_state,
    )

    # Why: returning these four objects makes the dataset easy to reuse elsewhere.
    return member_texts, member_labels, non_member_texts, non_member_labels


# Why: this block lets you run dataset.py directly for a quick sanity check without affecting imports.
if __name__ == "__main__":
    # Why: load the default dataset split so we can verify everything is working.
    member_texts, member_labels, non_member_texts, non_member_labels = load_data()

    # Why: these prints confirm the two groups were created with the expected sizes.
    print("Members:", len(member_texts))
    print("Non-members:", len(non_member_texts))

    # Why: these prints help verify that labels exist and the text looks correct.
    print("First member label:", member_labels[0])
    print("First member review preview:", member_texts[0][:200])