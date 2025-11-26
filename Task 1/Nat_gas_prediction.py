import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def get_price_estimate(input_date):
    """
    Estimates the natural gas price for a given date.
    
    Args:
        input_date (date or str): The date to estimate price for. 
                                  Can be a datetime.date object or string "YYYY-MM-DD".
    
    Returns:
        float: Estimated price.
    """
    # Load and process data
    # (In a real production environment, load/train once outside the function)
    df = pd.read_csv('Nat_Gas.csv')
    df['Dates'] = pd.to_datetime(df['Dates'])
    
    # Reference start date for the model
    start_date = df['Dates'].min()
    
    # Feature Engineering: Convert dates to days from start
    df['Days_From_Start'] = (df['Dates'] - start_date).dt.days
    
    # Create seasonal features (Sine and Cosine for 365.25 day cycle)
    df['sin_time'] = np.sin(2 * np.pi * df['Days_From_Start'] / 365.25)
    df['cos_time'] = np.cos(2 * np.pi * df['Days_From_Start'] / 365.25)
    
    # Fit the Linear Regression Model
    X = df[['Days_From_Start', 'sin_time', 'cos_time']]
    y = df['Prices']
    model = LinearRegression()
    model.fit(X, y)
    
    # Process input date
    if isinstance(input_date, str):
        input_date = pd.to_datetime(input_date)
    else:
        input_date = pd.to_datetime(input_date)
        
    days_from_start = (input_date - start_date).days
    
    # Create features for the input date
    input_sin = np.sin(2 * np.pi * days_from_start / 365.25)
    input_cos = np.cos(2 * np.pi * days_from_start / 365.25)
    
    # Predict
    input_features = pd.DataFrame([[days_from_start, input_sin, input_cos]], 
                                  columns=['Days_From_Start', 'sin_time', 'cos_time'])
    estimated_price = model.predict(input_features)[0]
    
    return estimated_price

# --- Visualization and Extrapolation Script ---
if __name__ == "__main__":
    # Load data for plotting
    df = pd.read_csv('Nat_Gas.csv')
    df['Dates'] = pd.to_datetime(df['Dates'])
    
    # Generate a range of dates for plotting (Historical + 1 Year Future)
    start_date = df['Dates'].min()
    end_date = df['Dates'].max() + timedelta(days=365)
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    estimated_prices = [get_price_estimate(d) for d in all_dates]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Dates'], df['Prices'], 'o', label='Actual Monthly Prices')
    plt.plot(all_dates, estimated_prices, '-', label='Estimated/Extrapolated Trend', alpha=0.7)
    
    plt.title('Natural Gas Price: Historical Estimate and 1-Year Extrapolation')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example Usage
    test_date = "2024-12-15"
    price = get_price_estimate(test_date)
    print(f"Estimated price for {test_date}: {price:.2f}")