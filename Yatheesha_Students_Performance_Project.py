import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
np.random.seed(42)
n = 100
study_hours = np.random.randint(1, 10, n)
attendance = np.random.randint(40, 100, n)
previous_score = np.random.randint(30, 100, n)
pass_fail = ((study_hours >= 6) &
             (attendance >= 75) &
              (previous_score >= 60)).astype(int)
data = {
    'Study_Hours' : study_hours,
    'Attendance' : attendance,
    'Previous_Score': previous_score,
    'Pass' : pass_fail
}
df = pd.DataFrame(data)
X = df[['Study_Hours', 'Attendance', 'Previous_Score']]
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_student = [[6, 85, 75]]  # Study Hours, Attendance, Previous Score
prediction = model.predict(new_student)
if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")

