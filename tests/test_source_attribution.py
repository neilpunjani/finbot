#!/usr/bin/env python3
"""Test source attribution functionality"""

from agents.langgraph_workflow import ChatbotWorkflow
import traceback

def test_source_attribution():
    """Test that source attribution is working properly"""
    try:
        print('🔄 Testing source attribution...')
        chatbot = ChatbotWorkflow()
        print('✅ Chatbot initialized')
        
        # Test queries for different sources
        test_queries = [
            "What columns are in the CSV data?",
            "Show me the budget data from Excel",
            "What tables are in the database?",
            "Check my recent emails"
        ]
        
        for query in test_queries:
            print(f'\n🔄 Testing: {query}')
            try:
                result = chatbot.process_query(query)
                print('✅ Response received')
                
                # Check if source attribution is present
                if '📊 **DATA SOURCE**:' in result:
                    print('✅ Source attribution found')
                    # Extract just the source line
                    lines = result.split('\n')
                    source_line = [line for line in lines if 'DATA SOURCE' in line][0]
                    print(f'   {source_line}')
                else:
                    print('❌ Source attribution missing')
                    
            except Exception as e:
                print(f'❌ Error: {e}')
                
    except Exception as e:
        print(f'❌ Error initializing chatbot: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    test_source_attribution()