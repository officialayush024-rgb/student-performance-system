
import joblib
import numpy as np
import pandas as pd

# Load the trained model
saved = joblib.load("model/student_model.pkl")
model = saved["model"]
features = saved["features"]

def predict_student(
    attendance,
    study_hours,
    previous_score,
    assignment_score,
    sleep_hours,
    participation
):
    student = pd.DataFrame([{
        "attendance_pct": attendance,
        "study_hours_per_day": study_hours,
        "previous_score": previous_score,
        "assignment_score": assignment_score,
        "sleep_hours": sleep_hours,
        "participation_level": participation
    }])

    prediction = model.predict(student[features])[0]
    prediction = float(np.clip(prediction, 0, 100))

    if prediction >= 80:
        category = "High"
    elif prediction >= 60:
        category = "Moderate"
    else:
        category = "Needs support"

    return round(prediction, 1), category


score, level = predict_student(
    attendance=85,
    study_hours=4,
    previous_score=70,
    assignment_score=80,
    sleep_hours=7,
    participation=4
)

print("Predicted score:", score)
print("Performance category:", level)
