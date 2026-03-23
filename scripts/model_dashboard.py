# scripts/model_dashboard.py

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import os

# ===============================
# 🔹 Set file paths
# ===============================
data_path = "data/raw/churn.csv"   # Ensure this CSV exists
output_path = "outputs/predictions_dashboard.csv"

# ===============================
# 🔹 Load dataset
# ===============================
st.title("📊 Telecom Customer Churn Prediction Dashboard")
st.write(f"Loading dataset: {data_path}")

if not os.path.exists(data_path):
    st.error(f"File not found: {data_path}")
    st.stop()

df = pd.read_csv(data_path)
st.write("Dataset Preview", df.head())

st.write("Columns in dataset:", df.columns)

# ===============================
# 🔹 Preprocessing
# ===============================
df = df.copy()

# Fill missing TotalCharges if any
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode target
le_churn = LabelEncoder()
df['Churn'] = le_churn.fit_transform(df['Churn'])

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns.drop('customerID')
df_encoded = pd.get_dummies(df.drop(['customerID'], axis=1), columns=categorical_cols)

# Features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ===============================
# 🔹 Models
# ===============================
# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# ===============================
# 🔹 Display Metrics
# ===============================
st.subheader("Model Metrics")

st.write("**Random Forest Metrics**")
st.write(f"Accuracy: {acc_rf:.3f}")
st.text(classification_report(y_test, y_pred_rf))

st.write("**Logistic Regression Metrics**")
st.write(f"Accuracy: {acc_lr:.3f}")
st.text(classification_report(y_test, y_pred_lr))

# ===============================
# 🔹 Top Features
# ===============================
st.subheader("Top Features (Random Forest)")
importances = rf.feature_importances_
feature_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_df = feature_df.sort_values(by='importance', ascending=False).head(15)

fig = px.bar(feature_df, x='importance', y='feature', orientation='h', title="Top 15 Features (Random Forest)")
st.plotly_chart(fig)

st.subheader("Top Features (Logistic Regression Coefficients)")
coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': lr.coef_[0]})
coef_df = coef_df.sort_values(by='coefficient', key=abs, ascending=False).head(15)
fig2 = px.bar(coef_df, x='coefficient', y='feature', orientation='h', title="Top 15 Features (Logistic Regression)")
st.plotly_chart(fig2)

# ===============================
# 🔹 Churn Distribution
# ===============================
st.subheader("Churn Distribution")
fig_churn = px.pie(df, names='Churn', title='Churn Distribution')
st.plotly_chart(fig_churn)

# ===============================
# 🔹 Numeric Feature Distributions
# ===============================
st.subheader("Numeric Feature Distributions")
for col in num_cols:
    fig_hist = px.histogram(df, x=col, color='Churn', barmode='overlay', title=f"{col} Distribution by Churn")
    st.plotly_chart(fig_hist)

# ===============================
# 🔹 Interactive Prediction Form
# ===============================
st.subheader("Predict Churn for New Customer")
with st.form(key='new_customer_form'):
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    input_df = pd.DataFrame([input_data])
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    pred_rf = rf.predict(input_df)[0]
    pred_prob_rf = rf.predict_proba(input_df)[0][1]
    pred_lr = lr.predict(input_df)[0]
    pred_prob_lr = lr.predict_proba(input_df)[0][1]

    st.write(f"**Random Forest Prediction:** {'Churn' if pred_rf==1 else 'No Churn'} (Probability: {pred_prob_rf:.2f})")
    st.write(f"**Logistic Regression Prediction:** {'Churn' if pred_lr==1 else 'No Churn'} (Probability: {pred_prob_lr:.2f})")

# ===============================
# 🔹 Save Predictions
# ===============================
output_df = X_test.copy()
output_df['Actual'] = y_test
output_df['RF_Predicted'] = y_pred_rf
output_df['LR_Predicted'] = y_pred_lr

os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
st.success(f"✅ Predictions saved at {output_path}")