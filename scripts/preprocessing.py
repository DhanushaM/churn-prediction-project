import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("✅ Data loaded successfully")
    return df

def clean_data(df):
    print("🧹 Cleaning data...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert numeric columns
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop missing values
    df = df.dropna()

    # Convert target column
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print("✅ Data cleaned successfully")
    return df

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"✅ Cleaned data saved at {path}")


if __name__ == "__main__":
    raw_path = "data/raw/churn.csv"
    processed_path = "data/processed/cleaned_churn.csv"

    df = load_data(raw_path)
    df = clean_data(df)
    save_data(df, processed_path)