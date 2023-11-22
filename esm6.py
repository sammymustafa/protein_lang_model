from setup import *

train_esm6 = torch.load("/content/train_esm6_data.pt")
test_esm6 = torch.load("/content/test_esm6_data.pt")

# COMPLETE HERE
# Train a classifier using whatever tool you prefer (simplest is sklearn's logisitc regression above!)
# When using LogisticRegression, set C=10 and max_iter=1000 for best results
# The data is provided as a list of tuples (vector, label)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

X_train_esm6, y_train_esm6 = zip(*train_esm6)
X_test_esm6, y_test_esm6 = zip(*test_esm6)

# Convert to numpy arrays
X_train_esm6 = np.array(X_train_esm6)
y_train_esm6 = np.array(y_train_esm6)
X_test_esm6 = np.array(X_test_esm6)
y_test_esm6 = np.array(y_test_esm6)

# Initialize the logistic regression model with the specified parameters
clf_esm6 = LogisticRegression(C=10, max_iter=1000)
clf_esm6.fit(X_train_esm6, y_train_esm6)

# Predict the class labels for the test set
y_pred_esm6 = clf_esm6.predict(X_test_esm6)

# Predict the probability scores for ROC-AUC for the positive class
y_score_esm6 = clf_esm6.predict_proba(X_test_esm6)[:, 1]

# Calculate accuracy
accuracy_esm6 = accuracy_score(y_test_esm6, y_pred_esm6)

# Calculate ROC-AUC
roc_auc_esm6 = roc_auc_score(y_test_esm6, y_score_esm6)

print(f"Accuracy for ESM-6 model: {accuracy_esm6}")
print(f"ROC-AUC for ESM-6 model: {roc_auc_esm6}")