# Student Performance Prediction using Machine Learning

## Project Overview
This project predicts student performance (Pass/Fail) using machine learning models.  
It analyzes various academic and behavioral factors to determine outcomes.

---

## Objective
- Predict student academic performance
- Compare multiple machine learning models
- Identify key factors affecting results

---

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset
- 1200 student records
- Features:
  - Study Hours
  - Attendance Percentage
  - Previous Marks
  - Assignment Score
  - Internal Marks
  - Parental Education
  - Internet Access
  - Extra Classes

Target:
- Final_Result (0 = Fail, 1 = Pass)

---

## Models Used
1. Decision Tree
2. Logistic Regression

---

## Methodology
1. Data preprocessing (encoding categorical variables)
2. Train-test split
3. Model training
4. Model evaluation using accuracy
5. Visualization of results

---

## Visualizations
- Confusion Matrix
- Correlation Heatmap
- Feature Importance
- Model Comparison Graph

---

## Output
- Accuracy comparison of models
- Prediction results CSV file
- Graphical analysis

---

## Project Structure
```
student-performance-ml/
│── data/
│── graphs/
│── outputs/
│── src/
│── README.md
│── requirements.txt
```

---

## How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the project:
```
python src/student_performance.py
```

---

## Results
- Decision Tree performs slightly better than Logistic Regression
- Key factors influencing performance include study hours, attendance, and previous marks

---

## Conclusion
Machine learning models can effectively predict student outcomes and help identify at-risk students.