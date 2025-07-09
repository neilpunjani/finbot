import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_excel():
    """Create a sample Excel file with multiple worksheets"""
    
    # Create sample_data directory if it doesn't exist
    os.makedirs('sample_data', exist_ok=True)
    
    # Sales Summary Sheet
    sales_summary = pd.DataFrame({
        'Month': ['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024', 'May 2024', 'Jun 2024'],
        'Total_Sales': [25000, 32000, 28000, 35000, 41000, 38000],
        'Units_Sold': [125, 160, 140, 175, 205, 190],
        'New_Customers': [15, 22, 18, 28, 35, 32],
        'Return_Rate': [0.05, 0.03, 0.04, 0.02, 0.03, 0.04],
        'Average_Order_Value': [200, 200, 200, 200, 200, 200]
    })
    
    # Budget Sheet
    budget_data = pd.DataFrame({
        'Department': ['Sales', 'Marketing', 'Operations', 'HR', 'IT', 'Finance'],
        'Q1_Budget': [50000, 30000, 40000, 25000, 35000, 20000],
        'Q1_Actual': [48000, 32000, 38000, 24000, 37000, 19000],
        'Q2_Budget': [55000, 35000, 42000, 26000, 38000, 22000],
        'Q2_Actual': [52000, 36000, 41000, 25000, 39000, 21000],
        'Q3_Budget': [58000, 38000, 45000, 28000, 40000, 24000],
        'Q3_Forecast': [56000, 37000, 44000, 27000, 41000, 23000]
    })
    
    # Product Performance Sheet
    product_performance = pd.DataFrame({
        'Product_ID': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
        'Product_Name': ['Laptop Pro', 'Wireless Mouse', 'Mechanical Keyboard', 'Ultra Monitor', 
                        'Noise-Cancel Headphones', 'Tablet Plus', 'Smartphone X', 'Wireless Charger'],
        'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 
                    'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'Units_Sold': [150, 450, 320, 180, 275, 95, 220, 380],
        'Revenue': [180000, 11250, 24000, 63000, 24750, 52250, 208800, 11400],
        'Cost': [120000, 6750, 16000, 42000, 16500, 34833, 139200, 7600],
        'Profit_Margin': [0.33, 0.40, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33],
        'Inventory_Level': [25, 150, 80, 45, 60, 20, 35, 120]
    })
    
    # Employee Data Sheet
    employee_data = pd.DataFrame({
        'Employee_ID': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008'],
        'Name': ['John Manager', 'Jane Analyst', 'Bob Developer', 'Alice Designer', 
                'Charlie Sales', 'Diana Marketing', 'Frank Support', 'Grace HR'],
        'Department': ['Sales', 'Finance', 'IT', 'Marketing', 'Sales', 'Marketing', 'Operations', 'HR'],
        'Salary': [75000, 65000, 80000, 70000, 60000, 68000, 55000, 72000],
        'Performance_Score': [4.2, 4.5, 4.8, 4.3, 4.1, 4.6, 4.0, 4.4],
        'Years_Experience': [8, 5, 10, 6, 4, 7, 3, 9],
        'Bonus_Eligible': [True, True, True, True, True, True, False, True]
    })
    
    # Regional Sales Sheet
    regional_sales = pd.DataFrame({
        'Region': ['North', 'South', 'East', 'West', 'Central'],
        'Q1_Sales': [45000, 38000, 52000, 41000, 34000],
        'Q2_Sales': [48000, 42000, 55000, 43000, 37000],
        'Q3_Sales': [52000, 45000, 58000, 46000, 39000],
        'Sales_Rep': ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
        'Target_Achievement': [0.95, 0.88, 1.05, 0.92, 0.86],
        'Customer_Count': [125, 108, 145, 118, 95]
    })
    
    # Create Excel file with multiple sheets
    excel_file = 'sample_data/financial_report.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        sales_summary.to_excel(writer, sheet_name='Sales_Summary', index=False)
        budget_data.to_excel(writer, sheet_name='Budget_Analysis', index=False)
        product_performance.to_excel(writer, sheet_name='Product_Performance', index=False)
        employee_data.to_excel(writer, sheet_name='Employee_Data', index=False)
        regional_sales.to_excel(writer, sheet_name='Regional_Sales', index=False)
    
    print(f"Created sample Excel file: {excel_file}")
    print("   Worksheets created:")
    print("   - Sales_Summary: Monthly sales data")
    print("   - Budget_Analysis: Department budget vs actual")
    print("   - Product_Performance: Product sales and profitability")
    print("   - Employee_Data: Employee information and performance")
    print("   - Regional_Sales: Sales performance by region")
    
    return excel_file

if __name__ == "__main__":
    create_sample_excel()