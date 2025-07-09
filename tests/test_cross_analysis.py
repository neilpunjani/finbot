#!/usr/bin/env python3
"""Test script for cross-agent analysis functionality"""

from agents.langgraph_workflow import ChatbotWorkflow
import traceback

def test_cross_analysis():
    """Test the cross-analysis functionality"""
    try:
        print('🔄 Initializing chatbot workflow...')
        chatbot = ChatbotWorkflow()
        print('✅ Workflow initialized successfully')
        
        # Test cross-analysis queries
        test_queries = [
            "Show me customer analysis using both csv and excel data",
            "Compare sales data across csv and excel files",
            "Analyze budget performance using both excel and csv data",
            "What insights can you find across all data sources?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f'\n🔄 Test {i}: {query}')
            try:
                result = chatbot.process_query(query)
                print(f'✅ Query processed successfully')
                print(f'Result type: {type(result)}')
                print(f'Result preview: {result[:300]}...')
            except Exception as e:
                print(f'❌ Error processing query: {e}')
                traceback.print_exc()
        
        print('\n✅ Cross-analysis testing completed')
        
    except Exception as e:
        print(f'❌ Error initializing workflow: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    test_cross_analysis()