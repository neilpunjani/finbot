#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

def test_imports():
    """Test if all imports work correctly"""
    try:
        from agents.router_agent import RouterAgent
        from agents.sql_agent import SQLAgent
        from agents.excel_agent import ExcelAgent
        from agents.csv_agent import CSVAgent
        from agents.email_agent import EmailAgent
        from agents.langgraph_workflow import ChatbotWorkflow
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_router_agent():
    """Test the router agent functionality"""
    print("\n🔍 Testing Router Agent...")
    try:
        router = RouterAgent()
        
        # Test queries
        test_queries = [
            "What is the total sales in the database?",
            "Analyze the Excel budget report",
            "Show me the CSV sales data",
            "Find emails about project updates"
        ]
        
        for query in test_queries:
            result = router.route_query(query)
            print(f"Query: '{query}' -> {result['data_source']} ({result['confidence']})")
        
        print("✅ Router agent working correctly")
        return True
    except Exception as e:
        print(f"❌ Router agent error: {e}")
        return False

def test_workflow_initialization():
    """Test if the workflow can be initialized"""
    print("\n🔄 Testing Workflow Initialization...")
    try:
        workflow = ChatbotWorkflow()
        status = workflow.get_system_status()
        print(status)
        print("✅ Workflow initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Workflow initialization error: {e}")
        return False

def test_query_processing():
    """Test basic query processing"""
    print("\n💬 Testing Query Processing...")
    try:
        workflow = ChatbotWorkflow()
        
        # Test with a simple routing query
        test_query = "What data sources are available?"
        response = workflow.process_query(test_query)
        print(f"Query: '{test_query}'")
        print(f"Response: {response[:200]}...")
        print("✅ Query processing working")
        return True
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        return False

def check_environment():
    """Check if required environment variables are set"""
    print("\n🔧 Checking Environment Configuration...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "EXCEL_FILE_PATH",
        "CSV_FILE_PATH",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Note: Some agents may not work without proper configuration")
    else:
        print("✅ All environment variables are set")
    
    return len(missing_vars) == 0

def main():
    """Run all tests"""
    print("🧪 Finaptive AI Chatbot - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Check", check_environment),
        ("Router Agent Test", test_router_agent),
        ("Workflow Initialization", test_workflow_initialization),
        ("Query Processing Test", test_query_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The chatbot is ready to use.")
        print("\nTo start the chatbot, run: python main.py")
    else:
        print("⚠️  Some tests failed. Check the configuration and dependencies.")
        if passed >= 3:
            print("   The core functionality should still work.")
    
    print("\nNext steps:")
    print("1. Copy .env.example to .env and fill in your API keys and file paths")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the chatbot: python main.py")

if __name__ == "__main__":
    main()