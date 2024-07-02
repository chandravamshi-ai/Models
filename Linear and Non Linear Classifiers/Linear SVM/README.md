### Understanding Multi-Class Classification with Linear SVM

In multi-class classification, the goal is to classify data points into one of several classes. While Linear SVMs are primarily binary classifiers, they can be extended to handle multi-class problems using different strategies.

#### Strategies for Multi-Class Classification with Linear SVM

1. **One-vs-Rest (OvR) / One-vs-All (OvA):**
   - **Approach:** Train one SVM classifier per class, where each classifier distinguishes one class from the rest.
   - **Process:**
     1. For a dataset with \( K \) classes, \( K \) binary classifiers are trained.
     2. Each classifier \( i \) is trained to separate class \( i \) from all other classes.
     3. During prediction, each classifier outputs a score, and the class with the highest score is selected.
   - **Example:**
     - For classes {A, B, C}, three classifiers are trained:
       - SVM1: A vs. {B, C}
       - SVM2: B vs. {A, C}
       - SVM3: C vs. {A, B}
     - For a new data point, the classifiers provide scores for A, B, and C, and the highest score determines the class.

2. **One-vs-One (OvO):**
   - **Approach:** Train one SVM classifier for every pair of classes.
   - **Process:**
     1. For a dataset with \( K \) classes, \( \frac{K(K-1)}{2} \) binary classifiers are trained.
     2. Each classifier \( (i, j) \) is trained to separate class \( i \) from class \( j \).
     3. During prediction, each classifier votes for one of the two classes, and the class with the most votes is selected.
   - **Example:**
     - For classes {A, B, C}, three classifiers are trained:
       - SVM1: A vs. B
       - SVM2: A vs. C
       - SVM3: B vs. C
     - For a new data point, each classifier votes for a class, and the class with the most votes wins.

#### Linear Decision Boundaries in Multi-Class Classification

- **One-vs-Rest:** Each SVM learns a hyperplane that separates one class from all others, resulting in \( K \) hyperplanes.
- **One-vs-One:** Each SVM learns a hyperplane that separates a pair of classes, resulting in \( \frac{K(K-1)}{2} \) hyperplanes.

#### Advantages and Disadvantages

- **One-vs-Rest (OvR):**
  - **Advantages:** Simpler to implement, requires fewer classifiers.
  - **Disadvantages:** Can be less accurate if one class is very similar to others.

- **One-vs-One (OvO):**
  - **Advantages:** More accurate for classes that are close together, each classifier is simpler (binary).
  - **Disadvantages:** Requires training more classifiers, more computationally intensive.

#### Practical Implementation

**Example using Scikit-Learn (Python):**

```python
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a One-vs-Rest Linear SVM
model_ovr = SVC(kernel='linear', decision_function_shape='ovr')
model_ovr.fit(X_train, y_train)

# Predict and evaluate
y_pred_ovr = model_ovr.predict(X_test)
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
print(f'One-vs-Rest Accuracy: {accuracy_ovr}')

# Train a One-vs-One Linear SVM
model_ovo = SVC(kernel='linear', decision_function_shape='ovo')
model_ovo.fit(X_train, y_train)

# Predict and evaluate
y_pred_ovo = model_ovo.predict(X_test)
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)
print(f'One-vs-One Accuracy: {accuracy_ovo}')
```

#### Summary

- **Linear SVMs** are inherently binary classifiers, but can be extended to multi-class classification using strategies like One-vs-Rest (OvR) and One-vs-One (OvO).
- **One-vs-Rest (OvR):** Trains one classifier per class against all other classes.
- **One-vs-One (OvO):** Trains one classifier for every pair of classes.
- Both strategies use multiple linear hyperplanes to separate classes, making linear SVMs versatile for multi-class problems.

