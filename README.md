# Heart Disease Prediction

## Project Overview
Heart disease is one of the leading causes of death worldwide. This project aims to predict the likelihood of heart disease using machine learning algorithms such as **Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN)**. By analyzing medical parameters such as **age, cholesterol levels, and heart rate**, we aim to build a predictive model that can assist in early diagnosis and prevention.

## Dataset Description
The dataset used in this project contains various medical attributes collected from patients. These features help determine the risk of heart disease.

### **Attributes in the Dataset:**
1. **Age** - Age of the patient in years.
2. **Sex** - Gender of the patient (1 = male, 0 = female).
3. **Chest Pain Type (cp)** - Types of chest pain experienced:
   - 1: Typical angina
   - 2: Atypical angina
   - 3: Non-anginal pain
   - 4: Asymptomatic
4. **Resting Blood Pressure (trestbps)** - Blood pressure in mm Hg at rest.
5. **Serum Cholesterol (chol)** - Cholesterol level in mg/dl.
6. **Fasting Blood Sugar (fbs)** - Whether blood sugar > 120 mg/dl (1 = True, 0 = False).
7. **Resting Electrocardiographic Results (restecg)** - ECG results (0, 1, 2).
8. **Maximum Heart Rate Achieved (thalach)** - Maximum heart rate recorded.
9. **Exercise-Induced Angina (exang)** - Whether angina occurred during exercise (1 = Yes, 0 = No).
10. **ST Depression Induced by Exercise (oldpeak)** - ST depression relative to rest.
11. **Slope of the Peak Exercise ST Segment (slope)** - Indicates the slope (0, 1, 2).
12. **Number of Major Vessels Colored by Fluoroscopy (ca)** - Ranges from 0 to 3.
13. **Thalassemia (thal)** - A blood disorder type (0, 1, 2, 3).
14. **Target (Output Variable)** - 1: Heart disease present, 0: No heart disease.

## Machine Learning Algorithms Used
### **1. Logistic Regression**
- A statistical method that predicts binary outcomes (heart disease: yes/no).
- Computes the probability of a patient having heart disease.
- Uses a sigmoid function to map predictions between 0 and 1.

### **2. Random Forest Classifier**
- An ensemble learning method that uses multiple decision trees.
- Improves prediction accuracy by averaging multiple tree results.
- Reduces overfitting compared to a single decision tree.

### **3. K-Nearest Neighbors (KNN)**
- A non-parametric algorithm that classifies based on the **k** nearest data points.
- Uses Euclidean distance to measure proximity.
- Works well with well-separated data.

## Model Evaluation
Each model is evaluated based on:
- **Accuracy** - The proportion of correct predictions.
- **Precision & Recall** - Measures the reliability of predictions.
- **Confusion Matrix** - Shows the performance of the classifier.
- **ROC Curve & AUC Score** - Evaluates the model’s ability to differentiate between classes.

## Results & Conclusion
- **Logistic Regression** achieved the highest accuracy and interpretability.
- **Random Forest** provided better performance but required more computational resources.
- **KNN** performed well but was sensitive to data distribution and required tuning.
- The final model is selected based on accuracy, precision, and recall scores.

This project demonstrates how machine learning can be leveraged in medical diagnostics. With further optimization and feature engineering, these models can assist in early heart disease detection, potentially saving lives.

## Installation & Usage
### **1. Install Dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **2. Run the Model**
```bash
python heart_disease_prediction.py
```

### **3. View Results**
The model outputs accuracy scores, confusion matrices, and visualizations comparing different classifiers.

## Future Improvements
- Use deep learning models like Neural Networks for better performance.
- Improve dataset size and quality for better generalization.
- Optimize feature selection for better model efficiency.

---
This project is intended for educational purposes and should not replace professional medical advice. For real-world applications, consult healthcare professionals and use validated medical models.



