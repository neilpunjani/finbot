import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType

class SQLAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is not set")
        
        self.db = SQLDatabase.from_uri(database_url)
        
        self.agent = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )
    
    def query(self, question: str) -> str:
        try:
            result = self.agent.run(question)
            return result
        except Exception as e:
            return f"Error querying database: {str(e)}"
    
    def get_schema_info(self) -> str:
        return self.db.get_table_info()
    
    def is_applicable(self, query: str) -> bool:
        sql_keywords = [
            'select', 'count', 'sum', 'average', 'max', 'min', 
            'group by', 'order by', 'where', 'join', 'table',
            'database', 'record', 'row', 'column', 'sql'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in sql_keywords)