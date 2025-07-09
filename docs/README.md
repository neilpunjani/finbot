# Finaptive AI Chatbot

An agentic AI chatbot solution that queries multiple data sources (SQL databases, Excel files, CSV files, and Outlook emails) using LangChain and LangGraph.

## Features

- **Multi-Source Querying**: Query SQL databases, Excel files, CSV files, and Outlook emails
- **Intelligent Routing**: Automatically determines which data source to query based on user input
- **Query Classification**: Classifies queries as analytical or informational
- **Multi-Worksheet Excel Support**: Analyzes and compares data across multiple Excel worksheets
- **Cross-Sheet Analysis**: Joins and compares data from different worksheets
- **Email Intelligence**: Searches and analyzes Outlook emails with AI-powered insights
- **Fallback Mechanisms**: Automatically tries alternative data sources if primary source fails
- **Error Handling**: Comprehensive error handling with helpful error messages

## Architecture

The solution uses a multi-agent architecture:

1. **Router Agent**: Determines which data source to query and classifies query type
2. **SQL Agent**: Handles database queries using LangChain SQL agents
3. **Excel Agent**: Processes Excel files with multi-worksheet support
4. **CSV Agent**: Analyzes CSV files using pandas
5. **Email Agent**: Searches and analyzes Outlook emails
6. **LangGraph Workflow**: Orchestrates all agents using a state-based workflow

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your configuration:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=your_database_connection_string_here
EMAIL_ADDRESS=your_email@company.com
EMAIL_PASSWORD=your_email_password_here
EXCEL_FILE_PATH=path/to/your/excel/file.xlsx

# CSV Configuration - Choose ONE method:
# Method 1: Directory with all CSV files (RECOMMENDED)
CSV_DIRECTORY=path/to/your/csv/directory

# Method 2: Comma-separated list of CSV files
# CSV_FILES=file1.csv,file2.csv,file3.csv,file4.csv

# Method 3: Individual CSV file paths
# CSV_FILE_PATH_CUSTOMERS=path/to/customers.csv
# CSV_FILE_PATH_SALES=path/to/sales.csv
```

### 3. Test the Setup

```bash
python test_chatbot.py
```

### 4. Run the Chatbot

```bash
python main.py
```

## Usage Examples

### SQL Database Queries
- "Show me the total sales for last month"
- "Count the number of customers in the database"
- "What are the top 5 products by revenue?"

### Excel File Analysis
- "Analyze the budget spreadsheet"
- "Compare data across different worksheets"
- "What's the total in the financial report?"

### CSV File Queries
- "Show me the summary of the sales data"
- "What are the trends in the customer data?"
- "Calculate the average from the CSV file"

### Email Analysis
- "Find emails about budget from last week"
- "Show me all emails from John Smith"
- "What are the main topics in my recent emails?"

## Data Sources

### SQL Database
- Supports PostgreSQL, MySQL, SQLite
- Uses LangChain SQL agents for natural language querying
- Handles complex queries with joins and aggregations

### Excel Files
- Supports multi-worksheet analysis
- Automatically detects relationships between worksheets
- Can join data across worksheets based on common columns
- Analyzes data types and suggests appropriate operations

### CSV Files
- **Multiple CSV Support**: Load unlimited CSV files from directory or individual paths
- **Intelligent File Selection**: Automatically selects the most relevant CSV file for each query
- **Pandas Agent Integration**: Uses AI-powered pandas agents for natural language queries
- **Flexible Configuration**: Directory-based, comma-separated list, or individual file paths
- **Statistical Analysis**: Provides comprehensive data insights and statistical analysis

### Outlook Emails
- Connects to live Outlook accounts via Exchange Web Services
- Searches emails by date, sender, subject, content
- Analyzes email patterns and trends
- Supports Outlook 365, Exchange Server, and Outlook.com

## System Commands

- `help` - Show available commands and examples
- `status` - Show system status and available agents
- `quit` or `exit` - Exit the chatbot

## Error Handling

The system includes comprehensive error handling:
- Connection failures are handled gracefully
- Missing data sources trigger fallback mechanisms
- Helpful error messages guide users to fix configuration issues
- Alternative agents are tried if primary agent fails

## Dependencies

- `langchain` - Core LangChain framework
- `langgraph` - Workflow orchestration
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools
- `langchain-experimental` - Experimental features
- `pandas` - Data analysis
- `openpyxl` - Excel file handling
- `exchangelib` - Outlook email integration
- `sqlalchemy` - Database connectivity
- `python-dotenv` - Environment variable management

## Requirements

- Python 3.12+
- OpenAI API key
- Access to data sources (database, files, email account)
- Network connectivity for email and database connections

## Troubleshooting

1. **Import Errors**: Run `pip install -r requirements.txt` to install dependencies
2. **Agent Initialization Failures**: Check your `.env` file configuration
3. **Database Connection Issues**: Verify your `DATABASE_URL` format
4. **Email Connection Problems**: Ensure email credentials are correct and 2FA is handled
5. **File Not Found Errors**: Check file paths in environment variables

## Future Enhancements

- Support for multiple databases, Excel files, and CSV files
- Web-based frontend interface
- Advanced analytics and visualization
- Integration with additional data sources
- Enhanced security features
- Caching and performance optimization