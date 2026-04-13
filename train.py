# Why: load_data gives us the prepared member and non-member groups from dataset.py.
from dataset import load_data

# Why: TfidfVectorizer converts raw text into numerical feature vectors the model can learn from.
from sklearn.feature_extraction.text import TfidfVectorizer

# Why: LogisticRegression is a simple and effective baseline model for text classification.
from sklearn.linear_model import LogisticRegression


# Why: this function handles the full target-model training workflow and returns everything needed later.
def train_target_model(member_size=500, non_member_size=500, random_state=42):
    # Why: load the controlled experiment data so the target model only trains on members.
    member_texts, member_labels, non_member_texts, non_member_labels = load_data(
        member_size=member_size,
        non_member_size=non_member_size,
        random_state=random_state,
    )

    # Why: the vectorizer learns a vocabulary from member data only, which avoids leakage from unseen data.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    # Why: fit_transform learns the vocabulary from member text and converts it into feature vectors.
    X_member = vectorizer.fit_transform(member_texts)

    # Why: transform applies the same learned vocabulary to non-member text without refitting.
    X_non_member = vectorizer.transform(non_member_texts)

    # Why: the target model is the classifier we will later attack.
    model = LogisticRegression(max_iter=1000, random_state=random_state)

    # Why: the model must only train on members for the membership labels to be meaningful.
    model.fit(X_member, member_labels)

    # Why: these scores show how differently the model behaves on seen versus unseen data.
    member_accuracy = model.score(X_member, member_labels)
    non_member_accuracy = model.score(X_non_member, non_member_labels)

    # Why: returning a dictionary keeps the code organized and makes attack.py easier to write.
    return {
        "model": model,
        "vectorizer": vectorizer,
        "member_texts": member_texts,
        "member_labels": member_labels,
        "non_member_texts": non_member_texts,
        "non_member_labels": non_member_labels,
        "X_member": X_member,
        "X_non_member": X_non_member,
        "member_accuracy": member_accuracy,
        "non_member_accuracy": non_member_accuracy,
    }


# Why: this block lets you run train.py directly to verify target-model training.
if __name__ == "__main__":
    # Why: train the model and capture all relevant outputs.
    results = train_target_model()

    # Why: print these values so you can inspect whether the target model behaves differently on members and non-members.
    print("Member accuracy:", results["member_accuracy"])
    print("Non-member accuracy:", results["non_member_accuracy"])