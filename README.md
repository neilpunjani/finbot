# Finaptive AI Chatbot

An agentic AI chatbot solution that queries multiple data sources (SQL databases, Excel files, CSV files, and Outlook emails) using LangChain and LangGraph.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   # Bash/Linux/macOS
   pip install -r requirements.txt
   ```
   ```powershell
   # PowerShell/Windows
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Bash/Linux/macOS
   cp .env.example .env
   # Edit .env with your settings
   ```
   ```powershell
   # PowerShell/Windows
   Copy-Item .env.example .env
   # Edit .env with your settings
   ```

3. **Test Setup**
   ```bash
   # Bash/Linux/macOS
   python tests/test_chatbot.py
   ```
   ```powershell
   # PowerShell/Windows
   python tests/test_chatbot.py
   ```

4. **Run Chatbot**
   ```bash
   # Bash/Linux/macOS
   python main.py
   ```
   ```powershell
   # PowerShell/Windows
   python main.py
   ```

## 📁 Project Structure

```
finaptive_chatbot/
├── 📁 src/                    # Core application code
│   ├── agents/                # AI agent implementations
│   ├── config/                # Configuration files
│   └── utils/                 # Utility functions
├── 📁 data/                   # All data files
│   ├── csv/                   # CSV files
│   ├── excel/                 # Excel files
│   └── samples/               # Sample/test data
├── 📁 tests/                  # All test files
├── 📁 scripts/                # Utility scripts
├── 📁 docs/                   # Documentation
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── .env                       # Environment configuration
```

## 🔧 Configuration

### CSV Files
Choose one configuration method:

```bash
# Method 1: Directory (Recommended)
CSV_DIRECTORY=data/csv

# Method 2: Comma-separated list
CSV_FILES=data/csv/file1.csv,data/csv/file2.csv

# Method 3: Individual paths
CSV_FILE_PATH_CUSTOMERS=data/csv/customers.csv
CSV_FILE_PATH_SALES=data/csv/sales.csv
```

### Other Data Sources
```bash
OPENAI_API_KEY=your_api_key_here
DATABASE_URL=your_database_url
EMAIL_ADDRESS=your_email@company.com
EMAIL_PASSWORD=your_email_password
EXCEL_FILE_PATH=data/excel/your_file.xlsx
```

## 🎯 Features

- **Multi-Source Querying**: Query SQL, Excel, CSV, and Outlook emails
- **Intelligent Routing**: Automatically selects the right data source
- **Natural Language**: Ask questions in plain English
- **Cross-Sheet Analysis**: Analyze data across multiple worksheets/files
- **Fallback Mechanisms**: Automatic retry with alternative sources
- **Source Attribution**: Track which data source answered each query

## 📚 Documentation

- [CSV Setup Guide](docs/CSV_SETUP_GUIDE.md) - Detailed CSV configuration
- [Full Documentation](docs/README.md) - Complete feature documentation

## 🧪 Testing

Run different test suites:

```bash
# Bash/Linux/macOS
# Complete system test
python tests/test_chatbot.py

# Environment validation
python scripts/debug_env.py

# Specific feature tests
python tests/test_csv_fix.py
python tests/test_source_attribution.py
```

```powershell
# PowerShell/Windows
# Complete system test
python tests/test_chatbot.py

# Environment validation
python scripts/debug_env.py

# Specific feature tests
python tests/test_csv_fix.py
python tests/test_source_attribution.py
```

## 🛠️ Development

### Adding New Data Sources
1. Create agent in `src/agents/`
2. Add to workflow in `src/agents/langgraph_workflow.py`
3. Update router in `src/agents/router_agent.py`

### Utility Scripts
- `scripts/debug_env.py` - Environment debugging
- `scripts/create_sample_excel.py` - Generate sample data

## 🔍 Debugging

For debugging issues:
1. Check environment configuration: `python scripts/debug_env.py`
2. Test individual agents: `python tests/test_[agent_name].py`
3. Check system status: Run chatbot and type `status`

## 📄 License

This project is part of the Finaptive AI system.