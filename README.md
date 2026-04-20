# Membership Inference Attack (MIA) Project Guide

## Overview

This project demonstrates a **membership inference attack (MIA)** on a machine learning model trained on text data.

A membership inference attack asks a very specific privacy question:

> Given a data sample and the model’s output on that sample, can we infer whether that sample was part of the model’s training set?

That question matters because a model may reveal private information **without directly revealing the training data itself**. If an attacker can determine that a specific person’s record was used in training, that can already be a privacy leak.

This guide is written for beginners. It explains:
- what **membership** means
- what each Python file does
- what each major block of code is doing
- why the project is structured this way
- how to talk about the project clearly in a presentation

---

# What “Membership” Means

In this project, **membership** means:

> whether a specific data point was used during the training of the target model

That gives every sample a binary label:

- `1` = **member**
- `0` = **non-member**


---

# Why Membership Matters

At first, this may seem like a small detail. But in real privacy-sensitive applications, membership can reveal important facts.

For example:

- If a model was trained on hospital records, membership might reveal whether a person’s medical record was included.
- If a model was trained on private emails, membership might reveal whether someone’s email was used.
- If a model was trained on legal or financial documents, membership might reveal whether a specific document was part of training.

That is why membership inference is considered a **privacy attack**.

The model does not need to reveal the full text of the private record for privacy harm to happen. Simply revealing whether that record was included can already be sensitive.

---

# Simple Analogy

Imagine a student studies 500 flashcards before a quiz.

Later, you show the student 1,000 flashcards:
- 500 that they studied
- 500 that they never saw

If the student answers some cards with much higher confidence, you may guess:

> “That was probably part of the study set.”

That is exactly the idea behind membership inference.

Translate the analogy like this:

- **Student** = trained model
- **Studied flashcards** = training samples
- **New flashcards** = unseen samples
- **Confidence of answer** = attack signal

The attacker is trying to detect whether the model behaves differently on things it has seen before.

---

# Project Structure

This project is organized into three files:

1. `dataset.py`
2. `train.py`
3. `attack.py`

Each file has a specific purpose:

- `dataset.py` prepares the dataset
- `train.py` trains the target model
- `attack.py` performs the privacy attack

This is a good design because each file has one main job. That makes the code easier to understand, test, explain, and present.

---

# Full Pipeline

The experiment works like this:

1. Load the IMDB dataset
2. Create two groups:
   - members
   - non-members
3. Convert text into numbers using TF-IDF
4. Train a sentiment classifier on members only
5. Ask the classifier for probability outputs on both groups
6. Extract behavior-based attack signals
7. Use those signals to infer membership

So this project really involves **two tasks**:

## Task 1: Target model task
Predict sentiment:
- positive
- negative

## Task 2: Attack model task
Predict membership:
- member
- non-member

That is what makes the project more sophisticated than ordinary text classification.

---

# Why the IMDB Dataset Was Chosen

The IMDB dataset is a good fit because:

- it is text-based
- it already has labels
- it is easy to explain
- it comes with train/test splits
- it works well with standard text-classification tools
- it is large enough to sample from

Each IMDB example contains:
- review text
- sentiment label

The sentiment label is:
- `0` = negative
- `1` = positive

This gives the project a clean classification task while still allowing you to study privacy leakage.

---

# File 1: `dataset.py`

## Purpose

`dataset.py` is responsible for preparing the data.

Its job is to decide:
- which samples become members
- which samples become non-members

Without that split, the rest of the project cannot exist.

---

## Code

```python
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
```

---

## Detailed Explanation

### `from datasets import load_dataset`

This imports the Hugging Face dataset loader.

Why this matters:
- it saves you from downloading files manually
- it keeps the experiment reproducible
- it makes the setup cleaner for demonstrations

---

### `from sklearn.model_selection import train_test_split`

This imports a function that can split or sample data.

In this project, it is used to create smaller controlled subsets:
- 500 members
- 500 non-members

This keeps the experiment manageable and easier to explain.

---

### `def load_data(...)`

This function wraps all of the dataset logic into one reusable place.

That means:
- `train.py` can reuse it
- `attack.py` can indirectly reuse it through `train.py`

This is better than copying the same code into multiple files.

---

### `dataset = load_dataset("imdb")`

This downloads and loads the IMDB dataset.

The IMDB dataset already includes:
- `dataset["train"]`
- `dataset["test"]`

This is useful because it gives you a natural way to define:
- members from train
- non-members from test

---

### Extracting texts and labels

```python
train_texts_full = dataset["train"]["text"]
train_labels_full = dataset["train"]["label"]

test_texts_full = dataset["test"]["text"]
test_labels_full = dataset["test"]["label"]
```

Machine learning usually separates:
- **inputs** = the data the model reads
- **labels** = the correct answers

Here:
- inputs = review text
- labels = sentiment

---

### Creating the member set

```python
member_texts, _, member_labels, _ = train_test_split(...)
```

The member set is sampled from the training split.

These samples are members because the target model will actually train on them later.

That is why the word **membership** is meaningful here.

---

### `stratify=train_labels_full`

This preserves label balance.

That means your smaller subset should still have a reasonable balance of:
- positive reviews
- negative reviews

This helps keep the experiment fair and stable.

---

### Creating the non-member set

```python
non_member_texts, _, non_member_labels, _ = train_test_split(...)
```

The non-member set is sampled from the official test split.

These reviews are never shown to the model during training.

That makes them valid non-members.

---

### `return ...`

This returns the four objects needed by the next stage:
- member texts
- member labels
- non-member texts
- non-member labels

---

### `if __name__ == "__main__":`

This lets the file behave in two ways:
- as a script when run directly
- as a module when imported elsewhere

If you run:

```bash
python dataset.py
```

it will print:
- how many members you have
- how many non-members you have
- a small review preview

That is helpful for checking whether the data stage works.

---

# File 2: `train.py`

## Purpose

`train.py` trains the **target model**.

The target model is the model that will later be attacked.

In this project, the target model learns sentiment classification:
- positive review
- negative review

But from the privacy side, what really matters is this:

> Does the target model behave differently on training samples than on unseen samples?

If the answer is yes, that difference may create privacy leakage.

---

## Why text must be converted into numbers

A model cannot directly understand raw text.

For example, a model cannot directly process:

> "This movie was fantastic."

It needs numerical input.

That is why `train.py` uses **TF-IDF**.

---

## What TF-IDF does

TF-IDF stands for:
- Term Frequency
- Inverse Document Frequency

It converts each review into a vector of numbers based on the importance of words.

General intuition:
- words that appear frequently in one review may matter
- words that appear in almost every review may matter less

This gives the model a numeric representation of language.

---

## Code

```python
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
```

---

## Detailed Explanation

### `from dataset import load_data`

This imports the data-preparation function from `dataset.py`.

This is good project organization because:
- one file handles data
- one file handles training

---

### `from sklearn.feature_extraction.text import TfidfVectorizer`

This imports the text vectorizer.

Its job is to translate text into numbers.

Without this, the target model would have no numerical features to learn from.

---

### `from sklearn.linear_model import LogisticRegression`

This imports the target classifier.

Even though the name includes “regression,” logistic regression is widely used for classification tasks.

It is a strong baseline because it is:
- simple
- fast
- standard
- easy to explain in a presentation

---

### `member_texts, ... = load_data(...)`

This loads the member and non-member sets.

At this point:
- members are training data
- non-members are unseen data

That split is what gives the attack meaning.

---

### `vectorizer = TfidfVectorizer(...)`

This creates the TF-IDF processor.

#### `max_features=5000`
Keeps at most 5,000 vocabulary terms.

This helps:
- control model size
- keep memory use reasonable
- simplify the experiment

#### `stop_words="english"`
Removes very common words such as:
- the
- and
- is
- of

These words usually add little sentiment information.

---

### `X_member = vectorizer.fit_transform(member_texts)`

This line does two things:
1. learns the vocabulary from member data
2. converts member text into vectors

This is important because vocabulary learning should come only from training data.

---

### `X_non_member = vectorizer.transform(non_member_texts)`

This applies the already-learned vocabulary to non-member text.

Notice that it does **not** use `fit_transform`.

That is intentional.

Why:
- non-members should not influence the vocabulary
- this keeps the setup honest

---

### `model = LogisticRegression(...)`

This creates the target model.

#### `max_iter=1000`
Gives the optimizer enough iterations to converge.

#### `random_state=random_state`
Helps keep the run reproducible.

---

### `model.fit(X_member, member_labels)`

This trains the target model on members only.

That is why the membership labels remain meaningful.

If you accidentally trained on both members and non-members, the privacy experiment would break.

---

### `member_accuracy = model.score(...)`

This checks how accurate the model is on the same data it trained on.

This is usually high.

---

### `non_member_accuracy = model.score(...)`

This checks how accurate the model is on unseen data.

This is often lower than member accuracy.

That difference can reflect overfitting or memorization, which may increase privacy leakage.

---

# File 3: `attack.py`

## Purpose

`attack.py` performs the membership inference attack.

Its job is to use the target model’s behavior to predict whether a sample was:
- a member
- a non-member

This is the privacy stage of the project.

---

## Main idea

The target model never directly says:
- “this sample was in my training set”
- “this sample was not in my training set”

So the attacker must infer membership from how the model behaves on each sample.

In the updated version of this project, `attack.py` now includes **two attack approaches**:

1. a **baseline threshold attack** based on confidence
2. a **learned attack model** trained on several behavior-based signals

This makes the project stronger because you can compare a simple interpretable attack against a more sophisticated one.

---

## What the attack features mean

### Max confidence
The highest predicted probability among all classes.

This tells you how certain the model was about its final prediction.

### True-class confidence
The probability assigned to the correct label.

This is often more informative than max confidence because it focuses on the true class specifically.

### Loss
The negative log probability of the true class.

Lower loss means the model handled that sample more confidently and correctly.

### Entropy
A measure of overall uncertainty in the full probability distribution.

Lower entropy means the model was more certain.

### Correctness
Whether the model predicted the true label correctly.

This is a simple binary signal that can still help separate members from non-members.

---

## Updated `attack.py` Code

```python
# Why: this gives attack.py access to the trained target model and its feature matrices.
from train import train_target_model

# Why: numpy is used for efficient numeric operations on model outputs and derived attack features.
import numpy as np

# Why: pandas makes it easier to organize attack features into a readable table-like structure.
import pandas as pd

# Why: train_test_split is used to create a train/test split for the attack model itself.
from sklearn.model_selection import train_test_split

# Why: LogisticRegression is used here again as a simple attack model that predicts member vs non-member.
from sklearn.linear_model import LogisticRegression

# Why: these metrics provide a stronger evaluation than plain accuracy alone.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# Why: matplotlib is used to visualize the ROC curve so you can show separability of members vs non-members.
import matplotlib.pyplot as plt


# Why: this helper computes Shannon entropy, which measures how uncertain the model is about a prediction.
def compute_entropy(probabilities):
    # Why: clipping prevents log(0), which would be undefined and break the calculation.
    clipped_probabilities = np.clip(probabilities, 1e-12, 1.0)

    # Why: entropy is lower when the model is very certain and higher when the model is more uncertain.
    return -np.sum(clipped_probabilities * np.log(clipped_probabilities), axis=1)


# Why: this helper computes the negative log probability of the true class, which is the sample loss.
def compute_true_class_loss(probabilities, true_labels):
    # Why: clipping prevents log(0) and keeps the loss calculation numerically stable.
    clipped_probabilities = np.clip(probabilities, 1e-12, 1.0)

    # Why: this selects the predicted probability assigned to the true class for each sample.
    true_class_probabilities = clipped_probabilities[np.arange(len(true_labels)), true_labels]

    # Why: lower loss usually indicates the model handled that sample more confidently and correctly.
    return -np.log(true_class_probabilities)


# Why: this helper converts target-model behavior into attack features for one set of samples.
def build_attack_features(model, feature_matrix, true_labels, membership_label):
    # Why: predict_proba is the core signal source because it gives probability behavior for each sample.
    probabilities = model.predict_proba(feature_matrix)

    # Why: predict returns the final class label so we can see whether the target model was correct.
    predictions = model.predict(feature_matrix)

    # Why: max confidence is a common membership signal because training samples often receive stronger confidence.
    max_confidence = np.max(probabilities, axis=1)

    # Why: true-class confidence measures how much probability the model assigned to the correct label specifically.
    true_class_confidence = probabilities[np.arange(len(true_labels)), true_labels]

    # Why: loss is often stronger than plain confidence because it directly measures how well the model fits a sample.
    true_class_loss = compute_true_class_loss(probabilities, true_labels)

    # Why: entropy captures uncertainty and can reveal whether the model is more certain on members.
    entropy = compute_entropy(probabilities)

    # Why: correctness is a useful binary signal because members may be predicted correctly more often.
    correctness = (predictions == np.array(true_labels)).astype(int)

    # Why: every attack row must carry the true membership label so the attack model can learn from it.
    membership = np.full(len(true_labels), membership_label)

    # Why: packaging attack signals into a DataFrame makes the attack dataset easy to inspect and use.
    attack_df = pd.DataFrame(
        {
            "max_confidence": max_confidence,
            "true_class_confidence": true_class_confidence,
            "loss": true_class_loss,
            "entropy": entropy,
            "correctness": correctness,
            "membership": membership,
        }
    )

    return attack_df


# Why: this baseline attack gives you a simple, interpretable result before moving to the learned attack model.
def run_threshold_attack(member_attack_df, non_member_attack_df):
    # Why: these are the average confidence values for members and non-members, used to define a simple threshold.
    member_mean_confidence = member_attack_df["max_confidence"].mean()
    non_member_mean_confidence = non_member_attack_df["max_confidence"].mean()

    # Why: the midpoint between group means creates a basic decision boundary.
    threshold = (member_mean_confidence + non_member_mean_confidence) / 2

    # Why: combine both groups into a single dataset for evaluation of the threshold rule.
    combined_df = pd.concat([member_attack_df, non_member_attack_df], ignore_index=True)

    # Why: if confidence exceeds the threshold, the baseline attack guesses the sample is a member.
    predicted_membership = (combined_df["max_confidence"] >= threshold).astype(int)

    # Why: these are the ground-truth membership labels used to evaluate the baseline attack.
    true_membership = combined_df["membership"]

    # Why: returning multiple metrics makes the attack easier to present and discuss.
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(true_membership, predicted_membership),
        "precision": precision_score(true_membership, predicted_membership),
        "recall": recall_score(true_membership, predicted_membership),
        "f1": f1_score(true_membership, predicted_membership),
    }


# Why: this learned attack is more sophisticated because it uses several target-model signals together.
def run_learned_attack(attack_df, random_state=42):
    # Why: these are the attack features that the attack model will use to predict membership.
    X_attack = attack_df[
        ["max_confidence", "true_class_confidence", "loss", "entropy", "correctness"]
    ]

    # Why: this is the true membership label that the attack model is trying to learn.
    y_attack = attack_df["membership"]

    # Why: splitting the attack dataset prevents us from evaluating the attack model on the same rows it trained on.
    X_train, X_test, y_train, y_test = train_test_split(
        X_attack,
        y_attack,
        test_size=0.30,
        stratify=y_attack,
        random_state=random_state,
    )

    # Why: logistic regression is a simple but effective baseline attack classifier for tabular attack features.
    attack_model = LogisticRegression(max_iter=1000, random_state=random_state)

    # Why: the attack model learns patterns in confidence, loss, entropy, and correctness that separate members from non-members.
    attack_model.fit(X_train, y_train)

    # Why: predict gives hard membership labels for standard classification metrics.
    y_pred = attack_model.predict(X_test)

    # Why: predict_proba gives membership scores used for ROC-AUC, which is a stronger separability metric.
    y_scores = attack_model.predict_proba(X_test)[:, 1]

    # Why: ROC curve points let us visualize the tradeoff between true positive rate and false positive rate.
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # Why: these metrics give a more complete view of the attack's success.
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_scores),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": thresholds,
        "attack_model": attack_model,
    }


# Why: this helper draws the ROC curve so you can visually explain how well the learned attack separates the two classes.
def plot_roc_curve(fpr, tpr, roc_auc):
    # Why: creating a dedicated figure keeps the ROC visualization clean and presentation-ready.
    plt.figure(figsize=(8, 6))

    # Why: this line shows the actual ROC curve traced by varying the decision threshold.
    plt.plot(fpr, tpr, label=f"Learned Attack ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)

    # Why: this diagonal line represents random guessing and gives an easy visual baseline for comparison.
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guessing")

    # Why: axis labels make the math interpretation of the plot immediately clear.
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Why: the title tells the audience exactly what model and metric they are looking at.
    plt.title("ROC Curve for Learned Membership Inference Attack")

    # Why: the legend helps distinguish the learned attack from the random baseline.
    plt.legend(loc="lower right")

    # Why: the grid improves readability when discussing specific parts of the curve during presentation.
    plt.grid(True)

    # Why: show displays the plot window so you can immediately inspect or present the result.
    plt.show()


# Why: this function runs the full upgraded attack pipeline from target-model training to attack evaluation.
def run_membership_inference_experiment(member_size=500, non_member_size=500, random_state=42):
    # Why: the target model must exist before we can extract attack signals from its behavior.
    training_results = train_target_model(
        member_size=member_size,
        non_member_size=non_member_size,
        random_state=random_state,
    )

    # Why: these objects are needed to build attack features for both groups.
    model = training_results["model"]
    X_member = training_results["X_member"]
    X_non_member = training_results["X_non_member"]
    member_labels = np.array(training_results["member_labels"])
    non_member_labels = np.array(training_results["non_member_labels"])

    # Why: these rows describe how the target model behaves on member samples.
    member_attack_df = build_attack_features(
        model=model,
        feature_matrix=X_member,
        true_labels=member_labels,
        membership_label=1,
    )

    # Why: these rows describe how the target model behaves on non-member samples.
    non_member_attack_df = build_attack_features(
        model=model,
        feature_matrix=X_non_member,
        true_labels=non_member_labels,
        membership_label=0,
    )

    # Why: combining both groups creates the full attack dataset for the learned attack model.
    full_attack_df = pd.concat([member_attack_df, non_member_attack_df], ignore_index=True)

    # Why: the threshold attack gives you a simple baseline to compare against the learned attack.
    threshold_results = run_threshold_attack(member_attack_df, non_member_attack_df)

    # Why: the learned attack uses several features at once and is more sophisticated than the baseline rule.
    learned_results = run_learned_attack(full_attack_df, random_state=random_state)

    # Why: returning everything makes it easy to inspect results, save them, or present them later.
    return {
        "target_member_accuracy": training_results["member_accuracy"],
        "target_non_member_accuracy": training_results["non_member_accuracy"],
        "member_avg_confidence": member_attack_df["max_confidence"].mean(),
        "non_member_avg_confidence": non_member_attack_df["max_confidence"].mean(),
        "member_avg_loss": member_attack_df["loss"].mean(),
        "non_member_avg_loss": non_member_attack_df["loss"].mean(),
        "threshold_attack": threshold_results,
        "learned_attack": learned_results,
        "attack_dataframe_preview": full_attack_df.head(10),
    }


# Why: this block lets you run the complete experiment directly from the terminal.
if __name__ == "__main__":
    # Why: run the full upgraded membership inference experiment using the default controlled setup.
    results = run_membership_inference_experiment()

    # Why: these prints summarize how the target model behaves on seen versus unseen samples.
    print("=== Target Model Performance ===")
    print("Member accuracy:", results["target_member_accuracy"])
    print("Non-member accuracy:", results["target_non_member_accuracy"])
    print("Average member confidence:", results["member_avg_confidence"])
    print("Average non-member confidence:", results["non_member_avg_confidence"])
    print("Average member loss:", results["member_avg_loss"])
    print("Average non-member loss:", results["non_member_avg_loss"])
    print()

    # Why: these prints show the simple baseline threshold attack results.
    print("=== Baseline Threshold Attack ===")
    print("Threshold:", results["threshold_attack"]["threshold"])
    print("Accuracy:", results["threshold_attack"]["accuracy"])
    print("Precision:", results["threshold_attack"]["precision"])
    print("Recall:", results["threshold_attack"]["recall"])
    print("F1:", results["threshold_attack"]["f1"])
    print()

    # Why: these prints show the stronger learned attack model results.
    print("=== Learned Attack Model ===")
    print("Accuracy:", results["learned_attack"]["accuracy"])
    print("Precision:", results["learned_attack"]["precision"])
    print("Recall:", results["learned_attack"]["recall"])
    print("F1:", results["learned_attack"]["f1"])
    print("ROC-AUC:", results["learned_attack"]["roc_auc"])
    print()

    # Why: this preview lets you inspect what the attack dataset actually looks like.
    print("=== Attack Feature Preview ===")
    print(results["attack_dataframe_preview"])
    print()

    # Why: plotting the ROC curve gives a visual explanation of how well the learned attack separates members and non-members.
    plot_roc_curve(
        results["learned_attack"]["fpr"],
        results["learned_attack"]["tpr"],
        results["learned_attack"]["roc_auc"],
    )
```

---

## Detailed Explanation

### `from train import train_target_model`

This imports the target-model pipeline from `train.py`.

That keeps the attack stage separate from the training stage, which makes the project cleaner and easier to explain.

---

### Why NumPy is used

`numpy` is used for:
- vector math
- probability calculations
- clipping values for numerical stability
- selecting the true-class probability for each sample

It is especially important for the entropy and loss calculations.

---

### Why pandas is used

`pandas` stores the attack features in a table-like structure.

That makes it easier to:
- inspect the attack dataset
- print previews
- select columns for the learned attack model
- explain what each row represents

Each row corresponds to one sample, and each column is one attack signal.

---

### `compute_entropy(probabilities)`

This computes Shannon entropy:

$$
H(p) = -\sum_i p_i \log p_i
$$

Interpretation:
- low entropy = the model is very certain
- high entropy = the model is less certain

Examples:
- `[0.99, 0.01]` gives low entropy
- `[0.50, 0.50]` gives high entropy

Members often have lower entropy because the model may be more certain on samples it saw during training.

---

### `compute_true_class_loss(probabilities, true_labels)`

This computes the negative log probability of the true class:

$$
\text{loss}(x, y) = -\log p(y \mid x)
$$

Interpretation:
- low loss means the model gave high probability to the correct class
- high loss means the model gave low probability to the correct class

This is one of the strongest membership signals because members often have lower loss than non-members.

---

### `build_attack_features(...)`

This function converts target-model outputs into a structured attack dataset.

For each sample, it computes:
- `max_confidence`
- `true_class_confidence`
- `loss`
- `entropy`
- `correctness`
- `membership`

This is the bridge between the target model and the attack model.

Instead of attacking raw text directly, the attack works on how the model behaves.

---

### `run_threshold_attack(...)`

This is the simple baseline attack.

It works like this:
1. compute the average max confidence for members
2. compute the average max confidence for non-members
3. place the threshold halfway between those two means
4. predict “member” when a sample’s confidence is above the threshold

This attack is useful because it is easy to explain:

> If the target model is much more confident on seen samples, then confidence alone may reveal membership.

Even if this baseline is not perfect, it gives you an interpretable starting point.

---

### `run_learned_attack(...)`

This is the stronger attack.

Instead of relying on only one signal, it uses:
- max confidence
- true-class confidence
- loss
- entropy
- correctness

These features are split into attack-train and attack-test sets so the attack model is evaluated fairly.

The attack classifier is logistic regression, which predicts:
- `1` for member
- `0` for non-member

This function returns:
- accuracy
- precision
- recall
- F1
- ROC-AUC
- false positive rates
- true positive rates
- ROC thresholds
- the trained attack model itself

That gives both numeric evaluation and visualization support.

---

### Why ROC-AUC was added

Accuracy alone can be misleading.

ROC-AUC measures how well the attack separates members from non-members across many thresholds, not just one fixed decision cutoff.

Interpretation:
- `0.50` means roughly random guessing
- closer to `1.00` means stronger separation

This makes ROC-AUC one of the most important metrics in the upgraded version of the project.

---

### `plot_roc_curve(fpr, tpr, roc_auc)`

This helper visualizes the learned attack.

The ROC curve shows the relationship between:
- **True Positive Rate** on the y-axis
- **False Positive Rate** on the x-axis

The diagonal reference line represents random guessing.

If the learned attack curve rises well above the diagonal, that means the attack is separating members from non-members better than chance.

This is useful in presentations because it gives a visual explanation of privacy leakage.

---

### `run_membership_inference_experiment(...)`

This is the full experiment driver.

It performs the complete pipeline:
1. train the target model
2. build attack features for members
3. build attack features for non-members
4. combine both groups into one attack dataset
5. run the baseline threshold attack
6. run the learned attack
7. return all important results

This function is useful because it centralizes the experiment into one callable workflow.

---

### The `__main__` block

When `attack.py` is run directly, it:
1. runs the full experiment
2. prints target-model statistics
3. prints baseline threshold attack metrics
4. prints learned attack metrics
5. prints a preview of the attack dataset
6. plots the ROC curve

That means the script now provides both:
- numerical evidence
- visual evidence

---

# Threshold Attack vs Learned Attack

A strong way to present the updated project is to explain that it now contains **two attack styles**.

## 1. Threshold attack
A simple rule-based baseline:
- compute a confidence threshold
- guess member when confidence is above that threshold

This is easy to interpret and useful for showing that privacy leakage can appear even with a very simple attacker.

## 2. Learned attack
A second classifier trained on multiple attack features.

This attack is stronger because it combines:
- confidence
- true-class confidence
- loss
- entropy
- correctness

This gives a more realistic attack scenario and makes the experiment more complete.


