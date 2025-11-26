import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

class PriceModel:
    def __init__(self):
        self.model = None
        self.start_date = None

    def train(self, csv_path='Nat_Gas_task_2.csv'):
        """
        Trains the pricing model using the provided CSV file.
        """
        # Load data
        try:
            df = pd.read_csv(csv_path)
            df['Dates'] = pd.to_datetime(df['Dates'])
        except FileNotFoundError:
            print(f"Error: {csv_path} not found. Please ensure the data file is available.")
            return

        self.start_date = df['Dates'].min()
        
        # Feature Engineering
        df['Days_From_Start'] = (df['Dates'] - self.start_date).dt.days
        df['sin_time'] = np.sin(2 * np.pi * df['Days_From_Start'] / 365.25)
        df['cos_time'] = np.cos(2 * np.pi * df['Days_From_Start'] / 365.25)
        
        # Fit Linear Regression
        X = df[['Days_From_Start', 'sin_time', 'cos_time']]
        y = df['Prices']
        self.model = LinearRegression()
        self.model.fit(X, y)
        print("Model trained successfully.")

    def get_price_estimate(self, input_date):
        """
        Returns the estimated price for a specific date.
        """
        if self.model is None:
            raise Exception("Model not trained. Call train() first.")

        input_date = pd.to_datetime(input_date)
        days_from_start = (input_date - self.start_date).days
        
        sin_time = np.sin(2 * np.pi * days_from_start / 365.25)
        cos_time = np.cos(2 * np.pi * days_from_start / 365.25)
        
        # Create input dataframe for prediction
        features = pd.DataFrame([[days_from_start, sin_time, cos_time]], 
                                columns=['Days_From_Start', 'sin_time', 'cos_time'])
        return self.model.predict(features)[0]


def calculate_contract_value(injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, max_volume, storage_cost_daily, price_model):
    """
    Calculates the value of a natural gas storage contract.
    
    Assumptions:
    - 'injection_rate' and 'withdrawal_rate' are the volumes processed on each respective date.
    - Cash flow is calculated as (Sales Revenue - Purchase Cost - Storage Cost).
    - Storage cost is calculated daily based on the volume held.
    
    Args:
        injection_dates (list): List of date strings (e.g., '2023-01-01') for injection.
        withdrawal_dates (list): List of date strings for withdrawal.
        injection_rate (float): Volume injected per event.
        withdrawal_rate (float): Volume withdrawn per event.
        max_volume (float): Maximum storage capacity.
        storage_cost_daily (float): Cost to store 1 unit of gas for 1 day.
        price_model (PriceModel): Trained PriceModel instance.
        
    Returns:
        float: The Net Present Value (NPV) or total profit of the contract.
    """
    
    # 1. Organize all actions into a chronological timeline
    actions = []
    for d in injection_dates:
        actions.append({'date': pd.to_datetime(d), 'type': 'inject', 'amount': injection_rate})
    for d in withdrawal_dates:
        actions.append({'date': pd.to_datetime(d), 'type': 'withdraw', 'amount': withdrawal_rate})
        
    # Sort actions by date
    actions.sort(key=lambda x: x['date'])
    
    # 2. Iterate through the timeline
    current_volume = 0
    total_value = 0
    
    # Start tracking from the first action date
    if not actions:
        return 0.0
        
    current_date = actions[0]['date']
    
    print("\n--- Contract Activity Log ---")
    
    for action in actions:
        action_date = action['date']
        days_elapsed = (action_date - current_date).days
        
        # A. Deduct Storage Costs for the time elapsed since last action
        if days_elapsed > 0 and current_volume > 0:
            period_cost = current_volume * storage_cost_daily * days_elapsed
            total_value -= period_cost
            # print(f"  Storage Cost ({days_elapsed} days): -${period_cost:,.2f}")
        
        # Get market price for the action date
        price = price_model.get_price_estimate(action_date)
        
        # B. Execute Action
        if action['type'] == 'inject':
            # Check if we have space
            if current_volume + action['amount'] <= max_volume:
                volume_to_inject = action['amount']
                current_volume += volume_to_inject
                cost = volume_to_inject * price
                total_value -= cost
                print(f"[{action_date.date()}] Inject {volume_to_inject:,.0f} units @ ${price:.2f} | Storage: {current_volume:,.0f}")
            else:
                print(f"[{action_date.date()}] SKIPPED INJECTION: Storage full.")
                
        elif action['type'] == 'withdraw':
            # Check if we have gas
            if current_volume - action['amount'] >= 0:
                volume_to_withdraw = action['amount']
                current_volume -= volume_to_withdraw
                revenue = volume_to_withdraw * price
                total_value += revenue
                print(f"[{action_date.date()}] Withdraw {volume_to_withdraw:,.0f} units @ ${price:.2f} | Storage: {current_volume:,.0f}")
            else:
                print(f"[{action_date.date()}] SKIPPED WITHDRAWAL: Insufficient gas.")
                
        current_date = action_date

    # Final cleanup: Ensure we account for storage costs until the last action (already done in loop)
    # Note: If gas remains in storage after the last date, its value is technically an asset, 
    # but for a standard cash-settled contract, we might consider it lost or force a sale.
    # For this prototype, we just calculate the realized cash flow.
    
    return total_value

# --- Usage Example ---

if __name__ == "__main__":
    # 1. Initialize and Train Model
    pricer = PriceModel()
    pricer.train('Nat_Gas_task_2.csv')
    
    # 2. Define Contract Parameters
    # Scenario: Inject in Summer (June-Aug), Withdraw in Winter (Dec-Feb)
    injection_dates = ['2023-06-15', '2023-07-15', '2023-08-15']
    withdrawal_dates = ['2023-12-15', '2024-01-15', '2024-02-15']
    
    injection_rate = 100000  # units per injection date
    withdrawal_rate = 100000 # units per withdrawal date
    max_storage_volume = 500000 
    
    # Cost: Assume $1000 per month for 1M units -> roughly $0.000033 per unit per day
    # Let's use a slightly higher example cost: $0.0001 per unit per day
    storage_cost_per_unit_daily = 0.0001
    
    # 3. Calculate Value
    contract_value = calculate_contract_value(
        injection_dates, 
        withdrawal_dates, 
        injection_rate, 
        withdrawal_rate, 
        max_storage_volume, 
        storage_cost_per_unit_daily,
        pricer
    )
    
    print(f"\nTotal Contract Value: ${contract_value:,.2f}")