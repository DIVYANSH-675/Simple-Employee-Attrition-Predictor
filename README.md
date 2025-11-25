# Employee Attrition Analysis

This project studies employee data and tries to find patterns related to **attrition** (whether an employee stays in the company or leaves).

It uses Python and simple machine learning to:
- load and clean the data
- explore important features
- train models to predict attrition
- show easy-to-understand results

---

## 1. Project Files

- `analysis.py` – Main script that loads data, prepares it, trains models, and prints results.
- `attrition.csv` – Dataset containing employee information.
- `DA_report.pdf` – Report explaining analysis and results.

Place all files in the **same folder** before running.

---

## 2. Requirements

You need Python **3.8 or higher** and the libraries listed in `requirements.txt`.

### requirements.txt
```
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

---

## 3. How to Install

### Step 1: Go to the project folder
```bash
cd path/to/your/project
```

### Step 2: Create a virtual environment
```bash
python -m venv venv
```

### Step 3: Activate it
- Windows (PowerShell):
```bash
venv\Scripts\Activate.ps1
```
- macOS/Linux:
```bash
source venv/bin/activate
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 4. How to Run

Run the main script:
```bash
python analysis.py
```

The script will:
1. Read the dataset  
2. Clean and prepare data  
3. Train ML models  
4. Show accuracy, reports, and results  
5. Save output files (like predictions)  

---

## 5. Understanding the Project (Beginner-Friendly)

- **Dataset:** Each row = one employee. Columns = information like age, role, salary, etc.
- **Goal:** Predict if an employee will leave.
- **Analysis:** Finds patterns such as which factors lead to higher attrition.
- **Machine Learning:** Trains models to make predictions on new data.

---

## 6. Common Issues

- Missing module → Run:
```bash
pip install -r requirements.txt
```

- Python not found → Use:
```bash
py analysis.py
```

---

## 7. Possible Improvements

- Add more ML models (XGBoost, SVM)  
- Add feature importance graphs  
- Create a small web UI using Flask/Streamlit  
- Add more visualizations  

---

## 8. License

Free to use for learning and practice.
