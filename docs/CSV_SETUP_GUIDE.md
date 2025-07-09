# CSV Files Setup Guide

## ✅ New Dynamic CSV Loading System

The CSV agent now supports **unlimited CSV files** with intelligent selection. Here are 3 easy ways to configure:

## Method 1: Directory-Based (Recommended)

Set a directory containing all your CSV files:

```bash
# In your .env file
CSV_DIRECTORY=/path/to/your/csv/files
```

Example directory structure:
```
/data/csv/
├── customer_data.csv
├── sales_data.csv
├── product_catalog.csv
├── employee_records.csv
├── financial_reports.csv
└── inventory_data.csv
```

The system will automatically:
- ✅ Load all `.csv` files in the directory
- ✅ Create individual pandas agents for each file
- ✅ Generate clean names (e.g., "customer data", "sales data")
- ✅ Intelligently select the right file for each query

## Method 2: Comma-Separated Paths

List all CSV file paths in one environment variable:

```bash
# In your .env file
CSV_FILES=/path/to/customers.csv,/path/to/sales.csv,/path/to/products.csv,/path/to/employees.csv
```

## Method 3: Individual Environment Variables

For backward compatibility, you can still use individual variables:

```bash
# In your .env file
CSV_FILE_PATH_CUSTOMERS=/path/to/customer_data.csv
CSV_FILE_PATH_SALES=/path/to/sales_data.csv
CSV_FILE_PATH_PRODUCTS=/path/to/product_catalog.csv
CSV_FILE_PATH_EMPLOYEES=/path/to/employee_records.csv
```

## 🎯 Intelligent File Selection

The system automatically selects the right CSV file based on:

1. **Filename matching** - "customer data" file for customer queries
2. **Column name matching** - Files with "State" column for location queries
3. **Content analysis** - Files with customer-related columns for customer queries
4. **LLM fallback** - AI decides when heuristics don't work

## Examples

With 4 CSV files loaded, these queries work automatically:

```
"What customers are in Texas?" 
→ Uses customer_data.csv

"Show me top selling products"
→ Uses sales_data.csv or product_catalog.csv

"How many employees are in marketing?"
→ Uses employee_records.csv

"What's our revenue by region?"
→ Uses financial_reports.csv
```

## 🚀 Benefits

- ✅ **Unlimited CSV files** - Add as many as needed
- ✅ **Zero configuration** - Just point to directory
- ✅ **Intelligent routing** - AI selects the right file
- ✅ **No code changes** - Works with existing queries
- ✅ **Backward compatible** - Old environment variables still work

## Migration from Old System

If you're using the old hardcoded approach:

**Old way:**
```bash
CSV_FILE_PATH=/path/to/data.csv
```

**New way (choose one):**
```bash
# Option 1: Directory
CSV_DIRECTORY=/path/to/csv/files

# Option 2: Multiple files
CSV_FILES=/path/to/file1.csv,/path/to/file2.csv,/path/to/file3.csv,/path/to/file4.csv
```

The system will automatically handle the rest!