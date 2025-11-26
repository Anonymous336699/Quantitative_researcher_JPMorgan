import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_models(file_path):
    """
    Loads data, trains models, and prints comparative performance.
    Returns the trained Logistic Regression model (preferred for this task).
    """
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

    # 2. Preprocessing
    # Drop identifier column
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    
    X = df.drop(columns=['default'])
    y = df['default']

    # Split into Train and Test sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Model Training & Comparison
    
    # Model A: Logistic Regression (Baseline, Interpretable)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    
    # Model B: Random Forest (Non-linear, Complex)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # 4. Evaluation
    print("--- Model Evaluation ---")
    
    # Logistic Regression Metrics
    y_pred_log = log_reg.predict(X_test)
    y_prob_log = log_reg.predict_proba(X_test)[:, 1]
    acc_log = accuracy_score(y_test, y_pred_log)
    auc_log = roc_auc_score(y_test, y_prob_log)
    print(f"Logistic Regression -> Accuracy: {acc_log:.4f}, AUC: {auc_log:.4f}")

    # Random Forest Metrics
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    print(f"Random Forest       -> Accuracy: {acc_rf:.4f}, AUC: {auc_rf:.4f}")

    # Feature Importance (from Random Forest)
    print("\n--- Key Risk Drivers ---")
    importances = pd.DataFrame({'feature': X.columns, 'importance': rf_clf.feature_importances_})
    print(importances.sort_values('importance', ascending=False).head(3))

    return log_reg

def calculate_expected_loss(loan_properties, model, recovery_rate=0.10):
    """
    Calculates the expected loss of a loan.
    
    Formula: Expected Loss = Probability of Default (PD) * Exposure at Default (EAD) * (1 - Recovery Rate)
    
    Args:
        loan_properties (dict): Dictionary containing the borrower's financial details.
                                Required keys: 'credit_lines_outstanding', 'loan_amt_outstanding',
                                'total_debt_outstanding', 'income', 'years_employed', 'fico_score'.
        model: Trained scikit-learn classification model (must have predict_proba).
        recovery_rate (float): The proportion of the loan recovered after default (default 10%).
        
    Returns:
        float: The expected monetary loss.
    """
    # Convert input dictionary to DataFrame (features must be in correct order)
    features_df = pd.DataFrame([loan_properties])
    
    # 1. Estimate Probability of Default (PD)
    # predict_proba returns [prob_no_default, prob_default]
    pd_estimate = model.predict_proba(features_df)[0][1]
    
    # 2. Identify Exposure at Default (EAD)
    # In this case, it is the outstanding loan amount
    ead = loan_properties['loan_amt_outstanding']
    
    # 3. Calculate Loss Given Default (LGD)
    lgd = 1.0 - recovery_rate
    
    # 4. Calculate Expected Loss
    expected_loss = pd_estimate * ead * lgd
    
    return expected_loss

# --- Example Usage ---
if __name__ == "__main__":
    # Train the model
    model = train_models('Task 3 and 4_Loan_Data.csv')
    
    if model:
        # Define a new loan applicant
        new_loan = {
            'credit_lines_outstanding': 2,
            'loan_amt_outstanding': 5000.0,
            'total_debt_outstanding': 10000.0,
            'income': 60000.0,
            'years_employed': 5,
            'fico_score': 650
        }

        # Calculate Expected Loss
        loss = calculate_expected_loss(new_loan, model, recovery_rate=0.10)
        
        print(f"\n--- New Loan Assessment ---")
        print(f"Loan Amount: ${new_loan['loan_amt_outstanding']:,.2f}")
        print(f"Expected Loss: ${loss:,.2f}")