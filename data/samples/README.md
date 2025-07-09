# Sample Data Files

This folder contains sample data files for testing the Finaptive AI Chatbot.

## Files Included

### CSV Files

1. **sales_data.csv** - Sample sales transactions
   - Contains: Date, Product, Category, Sales Amount, Quantity, Customer ID, Region
   - 26 records spanning January-February 2024
   - Electronics products with sales data

2. **customer_data.csv** - Sample customer information
   - Contains: Customer ID, Name, Email, Phone, City, State, Country, Age, Registration Date, Customer Type
   - 24 customer records with demographics and registration info
   - Premium and Standard customer types

### Excel File

**financial_report.xlsx** - Multi-worksheet financial report (to be created)
- **Sales_Summary**: Monthly sales totals and KPIs
- **Budget_Analysis**: Department budgets vs actual spending
- **Product_Performance**: Product sales and profitability metrics
- **Employee_Data**: Employee information and performance scores
- **Regional_Sales**: Sales performance by geographic region

## Test Queries

### For CSV Files

**Sales Data Queries:**
- "What's the total sales amount in the CSV file?"
- "Which product has the highest sales?"
- "Show me sales by region"
- "What's the average quantity sold?"

**Customer Data Queries:**
- "How many premium customers do we have?"
- "What's the average age of our customers?"
- "Show me customers by state"
- "Which customers registered in 2024?"

### For Excel File

**Cross-Sheet Analysis:**
- "Compare sales performance with budget targets"
- "Which employees are in the top-performing regions?"
- "Show me product performance vs employee performance"
- "Calculate total revenue across all sheets"

**Individual Sheet Queries:**
- "What's the best performing month in sales?"
- "Which department is over budget?"
- "Show me the top 3 products by revenue"
- "What's the average performance score by department?"

## Environment Setup

Update your `.env` file with these sample file paths:

```
EXCEL_FILE_PATH=sample_data/financial_report.xlsx
CSV_FILE_PATH=sample_data/sales_data.csv
```

## Creating the Excel File

After installing dependencies, run:

```bash
python create_sample_excel.py
```

This will create the multi-worksheet Excel file with sample financial data.

## Data Relationships

The sample data includes relationships that can be explored:
- **Customer_ID** links sales_data.csv and customer_data.csv
- **Department** appears in both Budget_Analysis and Employee_Data sheets
- **Region** data can be cross-referenced between sheets
- **Product** performance can be analyzed across multiple dimensions

## Testing Strategy

1. Start with simple queries on individual files
2. Test cross-sheet analysis in Excel
3. Try joining CSV data using Customer_ID
4. Experiment with analytical vs informational queries
5. Test the router's ability to select the right data source