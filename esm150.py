from setup import *

train_esm150 = torch.load("/content/train_esm150_data.pt")
test_esm150 = torch.load("/content/test_esm150_data.pt")

X_train_esm150, y_train_esm150 = zip(*train_esm150)
X_test_esm150, y_test_esm150 = zip(*test_esm150)

# Convert to numpy arrays
X_train_esm150 = np.array(X_train_esm150)
y_train_esm150 = np.array(y_train_esm150)
X_test_esm150 = np.array(X_test_esm150)
y_test_esm150 = np.array(y_test_esm150)

# Initialize the logistic regression model with the specified parameters
clf_esm150 = LogisticRegression(C=10, max_iter=1000)
clf_esm150.fit(X_train_esm150, y_train_esm150)

# Predict the class labels for the test set
y_pred_esm150 = clf_esm150.predict(X_test_esm150)

# Predict the probability scores for ROC-AUC for the positive class
y_score_esm150 = clf_esm150.predict_proba(X_test_esm150)[:, 1]

# Calculate accuracy
accuracy_esm150 = accuracy_score(y_test_esm150, y_pred_esm150)

# Calculate ROC-AUC
roc_auc_esm150 = roc_auc_score(y_test_esm150, y_score_esm150)

print(f"Accuracy for ESM-150 model: {accuracy_esm150}")
print(f"ROC-AUC for ESM-150 model: {roc_auc_esm150}")