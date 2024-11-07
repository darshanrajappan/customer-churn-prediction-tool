import mysql.connector
import pandas as pd

def connect_to_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL user
        password="Vision1215@Mysql",  # Replace with your MySQL password
        database="customer_db"
    )
    return conn

def fetch_data():
    conn = connect_to_db()
    query = "SELECT * FROM customers"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(0, inplace=True)  # Fill missing values
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['TotalSpent'] = df['MonthlyCharges'] * df['tenure']
    return df

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def load_data_from_mysql():
    df = fetch_data()  # Fetch data from MySQL
    df = preprocess_data(df)
    messagebox.showinfo("Data Load", "Data loaded from MySQL")
    return df

def load_data_from_csv():
    file_path = filedialog.askopenfilename()
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    messagebox.showinfo("Data Load", "Data loaded from CSV")
    return df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Define the pipeline for preprocessing and training the model
def create_pipeline():
    numeric_features = ["tenure", "MonthlyCharges", "TotalSpent"]
    categorical_features = ["gender", "Partner", "Dependents", "PhoneService", 
                            "MultipleLines", "InternetService", "OnlineSecurity", 
                            "OnlineBackup", "DeviceProtection", "TechSupport", 
                            "StreamingTV", "StreamingMovies", "Contract", 
                            "PaperlessBilling", "PaymentMethod"]

    # Preprocessor: scale numeric features, one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features)
        ]
    )

    # Build the full pipeline with preprocessing and model training
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    return pipeline

def train_model(df):
    X = df.drop(columns=["Churn", "customerID"])  # Exclude target and ID columns
    y = df["Churn"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Get model accuracy on the test set
    accuracy = pipeline.score(X_test, y_test)
    return pipeline, accuracy

def train_and_predict(df):
    pipeline, accuracy = train_model(df)
    messagebox.showinfo("Model Training", f"Model trained with accuracy: {accuracy*100:.2f}%")
    return pipeline

def predict_churn(pipeline, customer_data):
    prediction = pipeline.predict(customer_data)
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    messagebox.showinfo("Prediction", f"The customer is likely to: {result}")

# Function to convert raw input into DataFrame format
def get_sample_customer():
    # Replace with actual customer data for prediction
    sample_customer = {
        "tenure": 5, 
        "MonthlyCharges": 65.6, 
        "TotalSpent": 328.0, 
        "gender": "Female", 
        "Partner": "No", 
        "Dependents": "Yes", 
        "PhoneService": "Yes", 
        "MultipleLines": "No", 
        "InternetService": "Fiber optic", 
        "OnlineSecurity": "No", 
        "OnlineBackup": "Yes", 
        "DeviceProtection": "No", 
        "TechSupport": "Yes", 
        "StreamingTV": "No", 
        "StreamingMovies": "Yes", 
        "Contract": "Month-to-month", 
        "PaperlessBilling": "Yes", 
        "PaymentMethod": "Credit card (automatic)"
    }
    
    # Convert to DataFrame for consistency
    customer_df = pd.DataFrame([sample_customer])
    return customer_df

def main():
    root = tk.Tk()
    root.title("Customer Churn Prediction Tool")

    df = None  # Initialize empty dataframe
    pipeline = None  # Initialize empty pipeline

    def load_mysql():
        nonlocal df
        df = load_data_from_mysql()

    def load_csv():
        nonlocal df
        df = load_data_from_csv()

    def train():
        nonlocal pipeline
        if df is not None:
            pipeline = train_and_predict(df)
        else:
            messagebox.showerror("Error", "Load data first")

    def predict():
        if pipeline is not None:
            customer_df = get_sample_customer()
            predict_churn(pipeline, customer_df)
        else:
            messagebox.showerror("Error", "Train the model first")

    # Buttons for loading data, training model, and predicting churn
    load_mysql_button = tk.Button(root, text="Load Data from MySQL", command=load_mysql)
    load_mysql_button.pack()

    load_csv_button = tk.Button(root, text="Load Data from CSV", command=load_csv)
    load_csv_button.pack()

    train_button = tk.Button(root, text="Train Model", command=train)
    train_button.pack()

    predict_button = tk.Button(root, text="Predict Churn", command=predict)
    predict_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
