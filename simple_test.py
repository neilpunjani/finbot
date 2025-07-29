#!/usr/bin/env python3
"""
Quick test of the RAG system pandas agent issue
"""
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set environment variables - you'll need to set these with your actual values
os.environ['OPENAI_API_KEY'] = 'your_openai_key_here'  # Replace with actual key
os.environ['EXCEL_FILE_PATH'] = '/mnt/c/Users/punja/OneDrive/Desktop/Work/Finaptive/Agentic_AI/finbot/finbot/data/excel/Finaptive PBI Mining Data Set.xlsx'

from src.agents.pure_workflow import PureAgenticWorkflow

def quick_test():
    """Quick test of the pandas agent debug"""
    
    print("🧪 **Quick RAG Test - Pandas Agent Debug**")
    print("=" * 50)
    
    try:
        # Check if we have the required environment variables
        if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_key_here':
            print("❌ Please set OPENAI_API_KEY environment variable")
            return
        
        # Initialize RAG workflow
        print("\n1. Initializing RAG Workflow...")
        workflow = PureAgenticWorkflow(use_rag=True)
        
        # Test a simple financial query
        print("\n2. Testing financial query...")
        query = "What is the cash ratio for 2024?"
        
        print(f"Query: {query}")
        print("-" * 40)
        
        result = workflow.process_query(query)
        
        print("\n✅ **Result:**")
        print(result)
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()