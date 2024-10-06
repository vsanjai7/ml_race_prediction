# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create or load your dataset (Big/Small, Odd/Even results)
# Example dataset (replace this with actual race data)
data = {
    'Race_Period': [2410060200, 2410060199, 2410060198, 2410060197, 2410060196],
    'Car1': [6, 5, 7, 4, 9],
    'Car2': [7, 6, 3, 8, 2],
    'Car3': [3, 9, 2, 1, 5],
    'Result_Type1': ['B', 'S', 'B', 'S', 'B'],  # Big/Small
    'Result_Type2': ['O', 'E', 'O', 'E', 'O']   # Odd/Even
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess data
# We will encode 'B', 'S', 'O', 'E' as numbers (1 for Big/Even, 0 for Small/Odd)
df['Result_Type1'] = df['Result_Type1'].map({'B': 1, 'S': 0})
df['Result_Type2'] = df['Result_Type2'].map({'O': 1, 'E': 0})

# Step 3: Define features (input) and labels (output)
# Using the car numbers as input features to predict Big/Small and Odd/Even outcomes
X = df[['Car1', 'Car2', 'Car3']]  # Features
y1 = df['Result_Type1']           # Labels for Big/Small
y2 = df['Result_Type2']           # Labels for Odd/Even

# Step 4: Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
_, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)

# Step 5: Train the Decision Tree Classifier for Big/Small prediction
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y1_train)

# Train another Decision Tree for Odd/Even prediction
clf2 = DecisionTreeClassifier()
clf2.fit(X_train, y2_train)

# Step 6: Make predictions on the test set
y1_pred = clf1.predict(X_test)  # Predict Big/Small
y2_pred = clf2.predict(X_test)  # Predict Odd/Even

# Step 7: Evaluate the accuracy of the models
accuracy1 = accuracy_score(y1_test, y1_pred)
accuracy2 = accuracy_score(y2_test, y2_pred)

print(f"Big/Small Prediction Accuracy: {accuracy1 * 100:.2f}%")
print(f"Odd/Even Prediction Accuracy: {accuracy2 * 100:.2f}%")

# Step 8: Predict the next race result
# Example: Predict the next result based on new car numbers (you can replace these with new inputs)
new_race_data = [[8, 3, 7]]  # New car data for prediction
big_small_pred = clf1.predict(new_race_data)
odd_even_pred = clf2.predict(new_race_data)

# Convert predictions back to labels
big_small_result = 'B' if big_small_pred[0] == 1 else 'S'
odd_even_result = 'O' if odd_even_pred[0] == 1 else 'E'

print(f"Predicted Result: {big_small_result} {odd_even_result}")
