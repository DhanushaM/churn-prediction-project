Telecom Customer Churn Prediction Dashboard

This project predicts customer churn for a telecom company using machine learning. It includes an interactive **Streamlit dashboard** where users can visualize data, analyze metrics, and predict churn for new customers.

Features

- Data Preprocessing: Cleaned and transformed telecom customer data.
- Machine Learning Models: 
  - Random Forest Classifier
  - Logistic Regression
- Advanced Features:
  - Feature importance analysis
  - Logistic regression coefficients
  - Churn distribution visualization
  - Numeric feature distributions
- Interactive Dashboard: Predict churn for new customers using input sliders and dropdowns.
- Outputs: Prediction results saved in `outputs/predictions_dashboard.csv`.

Dataset

- Dataset used: [Telecom Customer Churn Dataset](https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn)
- Stored in `data/raw/churn.csv`

Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/DhanushaM/churn-prediction-project.git
   cd churn-prediction-project

2.Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux

3. Install dependencies:
pip install -r requirements.txt
Usage

4. Run the model script:
python scripts/model.py

5.Run the interactive dashboard:
streamlit run scripts/model_dashboard.py

Folder Structure
churn-prediction-project/
│
├─ data/
│  └─ raw/churn.csv
│
├─ outputs/
│  └─ predictions_dashboard.csv
│
├─ scripts/
│  ├─ model.py
│  ├─ model_dashboard.py
│  ├─ preprocessing.py
│  └─ utils.py
│
├─ requirements.txt
└─ README.md
