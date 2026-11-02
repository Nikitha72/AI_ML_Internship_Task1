# Task 4 – Student Performance Prediction

## Files
- `Task4_End_to_End_ML_Project.ipynb` – complete notebook workflow.
- `app.py` – optional Streamlit prediction interface.
- `requirements.txt` – Python dependencies.
- `student-mat.csv` – download separately from the UCI Student Performance dataset.
- `best_student_model.joblib`, CSV files and charts – generated when the notebook runs.

## Dataset
UCI Student Performance dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

Download the Maths file named `student-mat.csv` and place it in the same folder as the notebook.

## Setup in VS Code PowerShell
```powershell
python -m pip install -r requirements.txt
```

Open the notebook in VS Code, select your Python environment/kernel, and run all cells from top to bottom.

## Run the optional Streamlit app
After successfully running the notebook:
```powershell
python -m streamlit run app.py
```

Keep `app.py` and `best_student_model.joblib` in the same folder.

## Target definition and limitations
The notebook defines Pass as final Maths grade G3 >= 10. It excludes G1 and G2 from model features to reduce target leakage for an early-warning scenario. The prediction is an estimate, not a guarantee. Report the actual metrics produced by your run; do not invent or pre-fill model scores.
