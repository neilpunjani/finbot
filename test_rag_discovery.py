#!/usr/bin/env python3
"""
Test script for RAG-enhanced discovery system
"""
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True, verbose=False)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.pure_workflow import PureAgenticWorkflow

def test_rag_system():
    """Test the RAG-enhanced discovery system"""
    
    print("🧪 **RAG Discovery System Test**")
    print("=" * 50)
    
    try:
        # Initialize RAG-enhanced workflow
        print("\n1. Initializing RAG-Enhanced Workflow...")
        workflow = PureAgenticWorkflow(use_rag=True, rebuild_index=True)
        
        # Test system status
        print("\n2. Testing System Status...")
        status = workflow.get_system_status()
        print(status)
        
        # Test available commands
        print("\n3. Testing Available Commands...")
        commands = workflow.get_available_commands()
        print(commands)
        
        # Test queries
        test_queries = [
            "What is the cash ratio for 2024?",
            "Calculate current ratio",
            "What was the revenue for Ontario?",
            "Show me gold production in 2023",
            "What were the training hours for mining operations?"
        ]
        
        print(f"\n4. Testing {len(test_queries)} Sample Queries...")
        print("-" * 40)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n**Test Query {i}**: {query}")
            print("-" * 30)
            
            try:
                result = workflow.process_query(query)
                print(f"✅ Result: {result[:200]}..." if len(result) > 200 else f"✅ Result: {result}")
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
        
        print("\n🎉 **RAG System Test Complete!**")
        
    except Exception as e:
        print(f"❌ **RAG System Test Failed**: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure OPENAI_API_KEY is set in .env file")
        print("2. Verify EXCEL_FILE_PATH points to valid Excel file")
        print("3. Check CSV_DIRECTORY contains CSV files")
        print("4. Install required dependencies: pip install -r requirements.txt")

def test_legacy_fallback():
    """Test fallback to legacy system"""
    
    print("\n🔄 **Testing Legacy Fallback**")
    print("=" * 40)
    
    try:
        # Initialize legacy workflow
        print("Initializing Legacy Workflow...")
        workflow = PureAgenticWorkflow(use_rag=False)
        
        # Test one query
        result = workflow.process_query("What is the cash ratio?")
        print(f"✅ Legacy system working: {result[:100]}...")
        
    except Exception as e:
        print(f"❌ Legacy fallback failed: {str(e)}")

def compare_performance():
    """Compare RAG vs Legacy performance (rough timing)"""
    
    print("\n⚡ **Performance Comparison**")
    print("=" * 35)
    
    import time
    
    query = "What is the cash ratio for 2024?"
    
    try:
        # Test RAG performance
        print("Testing RAG performance...")
        rag_workflow = PureAgenticWorkflow(use_rag=True)
        
        start_time = time.time()
        rag_result = rag_workflow.process_query(query)
        rag_time = time.time() - start_time
        
        print(f"✅ RAG Time: {rag_time:.2f} seconds")
        
        # Test Legacy performance
        print("Testing Legacy performance...")
        legacy_workflow = PureAgenticWorkflow(use_rag=False)
        
        start_time = time.time()
        legacy_result = legacy_workflow.process_query(query)
        legacy_time = time.time() - start_time
        
        print(f"✅ Legacy Time: {legacy_time:.2f} seconds")
        
        # Calculate improvement
        if legacy_time > 0:
            improvement = ((legacy_time - rag_time) / legacy_time) * 100
            print(f"🚀 Performance Improvement: {improvement:.1f}%")
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 **RAG Discovery System Testing Suite**")
    print("=" * 60)
    
    # Run tests
    test_rag_system()
    test_legacy_fallback()
    compare_performance()
    
    print("\n✅ **All Tests Complete!**")
    print("\nNext Steps:")
    print("1. Review test results above")
    print("2. If successful, RAG system is ready for production")
    print("3. If issues found, check environment configuration")
    print("4. Use 'rebuild_rag_index()' method if data sources change")