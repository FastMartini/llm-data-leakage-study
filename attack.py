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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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

    # Why: these metrics give a more complete view of the attack's success.
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_scores),
        "attack_model": attack_model,
    }


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