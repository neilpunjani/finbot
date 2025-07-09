#!/usr/bin/env python3
"""Test the fixed CSV agent"""

from agents.csv_agent import CSVAgent
import traceback

def test_csv_agent():
    try:
        print('🔄 Testing CSV agent...')
        csv_agent = CSVAgent()
        print('✅ CSV agent initialized')
        
        # Test queries
        test_queries = [
            "What columns are in the customer data?",
            "Show me customer analysis",
            "What are the top 5 customers by sales amount?",
            "Give me a summary of the sales data"
        ]
        
        for query in test_queries:
            print(f'\n🔄 Testing: {query}')
            try:
                result = csv_agent.query(query)
                print(f'✅ Result: {result[:200]}...')
            except Exception as e:
                print(f'❌ Error: {e}')
                
    except Exception as e:
        print(f'❌ Error initializing CSV agent: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_agent()