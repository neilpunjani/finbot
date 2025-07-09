import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
try:
    from exchangelib import Credentials, Account, DELEGATE, Message, Folder
    EXCHANGELIB_AVAILABLE = True
except ImportError:
    EXCHANGELIB_AVAILABLE = False

class EmailAgent:
    def __init__(self):
        if not EXCHANGELIB_AVAILABLE:
            raise ImportError("exchangelib is not available. Install it with: pip install exchangelib")
            
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        email_address = os.getenv("EMAIL_ADDRESS")
        email_password = os.getenv("EMAIL_PASSWORD")
        
        if not email_address or not email_password:
            print("Warning: EMAIL_ADDRESS and EMAIL_PASSWORD not configured. Email agent will not be fully functional.")
            self.account = None
            return
        
        try:
            credentials = Credentials(email_address, email_password)
            self.account = Account(
                primary_smtp_address=email_address,
                credentials=credentials,
                autodiscover=True,
                access_type=DELEGATE
            )
            print(f"Successfully connected to email account: {email_address}")
        except Exception as e:
            print(f"Warning: Failed to connect to email account: {str(e)}")
            self.account = None
    
    def search_emails(self, query: str, days_back: int = 30, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search emails based on query parameters"""
        if not self.account:
            return []
            
        try:
            # Calculate date range
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Search in inbox
            emails = []
            for item in self.account.inbox.filter(
                datetime_received__gte=start_date
            ).order_by('-datetime_received')[:max_results]:
                
                if isinstance(item, Message):
                    email_data = {
                        'subject': item.subject or 'No Subject',
                        'sender': str(item.sender) if item.sender else 'Unknown Sender',
                        'received': item.datetime_received.strftime('%Y-%m-%d %H:%M:%S') if item.datetime_received else 'Unknown Date',
                        'body': str(item.body)[:1000] if item.body else '',  # Truncate body and ensure string
                        'has_attachments': bool(item.has_attachments),
                        'importance': str(item.importance) if item.importance else 'Normal',
                        'categories': list(item.categories) if item.categories else []
                    }
                    emails.append(email_data)
            
            return emails
            
        except Exception as e:
            raise Exception(f"Error searching emails: {str(e)}")
    
    def analyze_emails_with_llm(self, emails: List[Dict[str, Any]], query: str) -> str:
        """Use LLM to analyze emails and answer the query"""
        if not emails:
            return "No emails found matching the search criteria."
        
        # Prepare email data for LLM analysis
        email_summaries = []
        for i, email in enumerate(emails[:20]):  # Limit to 20 emails for context
            summary = f"""
            Email {i+1}:
            Subject: {email['subject']}
            From: {email['sender']}
            Date: {email['received']}
            Body Preview: {email['body'][:200]}...
            Has Attachments: {email['has_attachments']}
            """
            email_summaries.append(summary)
        
        analysis_prompt = f"""
        I have {len(emails)} emails to analyze. Here are the details of the most recent ones:
        
        {chr(10).join(email_summaries)}
        
        User Query: {query}
        
        Please analyze these emails and provide insights to answer the user's question. 
        Consider patterns, trends, key information, and any relevant details from the emails.
        If the query is about specific content, search through the email subjects and bodies.
        """
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        return response.content
    
    def query(self, question: str) -> str:
        """Main query method that searches emails and analyzes them"""
        if not self.account:
            return "Email agent is not connected. Please check your email configuration in the .env file."
            
        try:
            # Extract search parameters from the question
            days_back = 30  # Default
            max_results = 50  # Default
            
            # Simple parameter extraction
            if "last week" in question.lower():
                days_back = 7
            elif "last month" in question.lower():
                days_back = 30
            elif "last year" in question.lower():
                days_back = 365
            
            # Search emails
            emails = self.search_emails(question, days_back, max_results)
            
            # Analyze with LLM
            analysis = self.analyze_emails_with_llm(emails, question)
            
            return f"Found {len(emails)} emails. Analysis:\n{analysis}"
            
        except Exception as e:
            return f"Error querying emails: {str(e)}"
    
    def get_email_stats(self) -> str:
        """Get basic email statistics"""
        if not self.account:
            return "Email agent is not connected. Please check your email configuration."
            
        try:
            # Get recent emails count
            recent_emails = list(self.account.inbox.filter(
                datetime_received__gte=datetime.now() - timedelta(days=7)
            ))
            
            # Get folder info
            folders = []
            try:
                for folder in self.account.root.walk():
                    if isinstance(folder, Folder):
                        folders.append(f"{folder.name}: {folder.total_count} items")
            except Exception:
                folders = ["Unable to retrieve folder information"]
            
            stats = f"Recent emails (last 7 days): {len(recent_emails)}\n"
            stats += "Available folders:\n" + "\n".join(folders[:10])  # Limit to 10 folders
            
            return stats
            
        except Exception as e:
            return f"Error getting email stats: {str(e)}"
    
    def is_applicable(self, query: str) -> bool:
        """Determine if this agent should handle the query"""
        email_keywords = [
            'email', 'emails', 'inbox', 'message', 'messages',
            'outlook', 'exchange', 'mail', 'sender', 'recipient',
            'subject', 'attachment', 'sent', 'received'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in email_keywords)