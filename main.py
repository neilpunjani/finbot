from dotenv import load_dotenv
from src.agents.langgraph_workflow import ChatbotWorkflow

load_dotenv()

def main():
    print("Initializing Finaptive AI Chatbot...")
    
    try:
        workflow = ChatbotWorkflow()
        print("✅ Chatbot initialized successfully!")
        print("\n" + workflow.get_system_status())
        print("\n" + "="*60)
        print("Finaptive AI Chatbot - Multi-Source Query System")
        print("="*60)
        print("Ask me questions about your data sources (SQL, Excel, CSV, or Emails)")
        print("Type 'help' for examples, 'status' for system info, or 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['help', 'examples']:
                print(workflow.get_available_commands())
                continue
                
            if user_input.lower() in ['status', 'system']:
                print(workflow.get_system_status())
                continue
                
            try:
                print("Processing your query...")
                response = workflow.process_query(user_input)
                print(f"\nAssistant: {response}\n")
                print("-" * 60 + "\n")
            except Exception as e:
                print(f"Error: {str(e)}\n")
                
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {str(e)}")
        print("\nPlease check your configuration:")
        print("1. Ensure all required environment variables are set in .env file")
        print("2. Verify data source files exist and are accessible")
        print("3. Check network connectivity for email/database connections")

if __name__ == "__main__":
    main()