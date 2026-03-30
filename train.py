# Why: we import the prepared member and non-member data from dataset.py instead of duplicating data logic.
from dataset import load_data

# Why: TfidfVectorizer converts raw review text into numerical features the classifier can learn from.
from sklearn.feature_extraction.text import TfidfVectorizer

# Why: LogisticRegression is a simple, fast baseline model for text classification.
from sklearn.linear_model import LogisticRegression


# Why: wrapping the training workflow in a function makes it reusable from attack.py.
def train_target_model():
    # Why: load the member and non-member sets for the controlled experiment.
    member_texts, member_labels, non_member_texts, non_member_labels = load_data()

    # Why: the vectorizer learns a vocabulary from member data only, which matches a real training setup.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    # Why: fit_transform learns the vocabulary on training members and converts them into feature vectors.
    X_member = vectorizer.fit_transform(member_texts)

    # Why: transform applies that same vocabulary to unseen non-member reviews.
    X_non_member = vectorizer.transform(non_member_texts)

    # Why: this is the target model that the membership inference attack will later probe.
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Why: the model must train only on member data so those samples truly count as members.
    model.fit(X_member, member_labels)

    # Why: these scores give a quick sanity check before moving to the attack stage.
    member_accuracy = model.score(X_member, member_labels)
    non_member_accuracy = model.score(X_non_member, non_member_labels)

    # Why: returning everything needed later keeps attack.py simple.
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


# Why: this block lets you run train.py directly to verify the target model trains correctly.
if __name__ == "__main__":
    # Why: train the model and capture all returned experiment objects.
    results = train_target_model()

    # Why: print accuracies so we can check whether the model behaves differently on members vs non-members.
    print("Member accuracy:", results["member_accuracy"])
    print("Non-member accuracy:", results["non_member_accuracy"])