CREATE DATABASE customer_db;

USE customer_db;

CREATE TABLE customers (
    customerID VARCHAR(255),
    gender VARCHAR(255),
    SeniorCitizen INT,
    Partner VARCHAR(255),
    Dependents VARCHAR(255),
    tenure INT,
    PhoneService VARCHAR(255),
    MultipleLines VARCHAR(255),
    InternetService VARCHAR(255),
    OnlineSecurity VARCHAR(255),
    OnlineBackup VARCHAR(255),
    DeviceProtection VARCHAR(255),
    TechSupport VARCHAR(255),
    StreamingTV VARCHAR(255),
    StreamingMovies VARCHAR(255),
    Contract VARCHAR(255),
    PaperlessBilling VARCHAR(255),
    PaymentMethod VARCHAR(255),
    MonthlyCharges FLOAT,
    TotalCharges FLOAT,
    Churn VARCHAR(255)
);
