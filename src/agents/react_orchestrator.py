from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
import os
import json
from dataclasses import dataclass
from enum import Enum

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ReasoningStep:
    thought: str
    action: str
    observation: str
    confidence: ConfidenceLevel

@dataclass
class AgentResponse:
    content: str
    confidence: ConfidenceLevel
    sources_used: List[str]
    reasoning_steps: List[ReasoningStep]
    needs_more_data: bool = False

class ReActOrchestrator:
    def __init__(self, csv_agent=None, excel_agent=None, sql_agent=None, email_agent=None):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Available tools
        self.tools = {
            'csv': csv_agent,
            'excel': excel_agent,
            'sql': sql_agent,
            'email': email_agent
        }
        
        # Conversation memory
        self.conversation_history = []
        self.reasoning_steps = []
        
    def process_query(self, query: str) -> AgentResponse:
        """Main ReAct reasoning loop"""
        
        # Initialize reasoning
        self.reasoning_steps = []
        max_iterations = 5
        
        for iteration in range(max_iterations):
            # THINK: Analyze current situation
            thought = self._think(query, iteration)
            
            # ACT: Decide what action to take
            action = self._act(thought, query)
            
            # OBSERVE: Execute action and get result
            observation = self._observe(action)
            
            # Store reasoning step
            step = ReasoningStep(
                thought=thought,
                action=action['type'],
                observation=observation['content'],
                confidence=observation['confidence']
            )
            self.reasoning_steps.append(step)
            
            # Check if we have enough information
            if self._should_finalize(observation, query):
                break
                
        # FINALIZE: Generate final response
        return self._finalize_response(query)
    
    def _think(self, query: str, iteration: int) -> str:
        """ReAct THINK phase - analyze situation and plan next step"""
        
        context = self._build_context(query, iteration)
        
        think_prompt = PromptTemplate.from_template("""
        You are an intelligent data analyst reasoning through a complex query.
        
        USER QUERY: {query}
        
        CURRENT CONTEXT:
        {context}
        
        AVAILABLE TOOLS:
        - CSV: Mining operations data (production, ESG, workforce, operational metrics)
        - EXCEL: Financial data (revenue, costs, balance sheet, P&L, trial balance)
        - SQL: Structured database with relational data
        - EMAIL: Email communications and correspondence
        
        REASONING STEP {iteration}:
        
        Think step by step about:
        1. What specific information do I need to answer this query?
        2. What have I learned so far from previous steps?
        3. What's the most logical next step?
        4. Do I need data from multiple sources?
        5. Am I confident in my understanding so far?
        
        Provide your reasoning in 2-3 sentences focusing on the next logical step.
        """)
        
        response = self.llm.invoke([
            HumanMessage(content=think_prompt.format(
                query=query,
                context=context,
                iteration=iteration + 1
            ))
        ])
        
        return response.content.strip()
    
    def _act(self, thought: str, query: str) -> Dict[str, Any]:
        """ReAct ACT phase - decide which tool to use and how"""
        
        action_prompt = PromptTemplate.from_template("""
        Based on your reasoning, decide what action to take next.
        
        REASONING: {thought}
        QUERY: {query}
        
        AVAILABLE ACTIONS:
        1. query_csv - Query CSV data for operational/mining metrics
        2. query_excel - Query Excel data for financial information
        3. query_sql - Query SQL database for structured data
        4. query_email - Query email communications
        5. finalize - I have enough information to provide final answer
        
        Respond with JSON format:
        {{
            "type": "action_name",
            "tool": "csv|excel|sql|email",
            "specific_query": "detailed question to ask the tool",
            "reasoning": "why this action makes sense"
        }}
        
        OR if you have enough information:
        {{
            "type": "finalize",
            "reasoning": "why I can now provide the final answer"
        }}
        """)
        
        response = self.llm.invoke([
            HumanMessage(content=action_prompt.format(
                thought=thought,
                query=query
            ))
        ])
        
        try:
            return json.loads(response.content.strip())
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "type": "query_csv",
                "tool": "csv",
                "specific_query": query,
                "reasoning": "JSON parsing failed, defaulting to CSV"
            }
    
    def _observe(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct OBSERVE phase - execute action and analyze result"""
        
        if action['type'] == 'finalize':
            return {
                "content": "Ready to finalize response",
                "confidence": ConfidenceLevel.HIGH,
                "data": None
            }
        
        # Execute the tool
        tool_name = action.get('tool')
        specific_query = action.get('specific_query', '')
        
        if tool_name not in self.tools or self.tools[tool_name] is None:
            return {
                "content": f"{tool_name} tool is not available",
                "confidence": ConfidenceLevel.LOW,
                "data": None
            }
        
        try:
            # Query the specific tool
            tool_result = self.tools[tool_name].query(specific_query)
            
            # Analyze the result for confidence and completeness
            analysis = self._analyze_tool_result(tool_result, specific_query)
            
            return {
                "content": tool_result,
                "confidence": analysis['confidence'],
                "data": analysis['data'],
                "completeness": analysis['completeness']
            }
            
        except Exception as e:
            return {
                "content": f"Error querying {tool_name}: {str(e)}",
                "confidence": ConfidenceLevel.LOW,
                "data": None
            }
    
    def _analyze_tool_result(self, result: str, query: str) -> Dict[str, Any]:
        """Analyze tool result for confidence and completeness"""
        
        analysis_prompt = PromptTemplate.from_template("""
        Analyze this tool result for confidence and completeness.
        
        ORIGINAL QUERY: {query}
        TOOL RESULT: {result}
        
        Evaluate:
        1. Confidence: How confident are you this result answers the query?
        2. Completeness: Is this a complete answer or partial?
        3. Data quality: Are there any obvious errors or inconsistencies?
        
        Respond with JSON:
        {{
            "confidence": "high|medium|low",
            "completeness": "complete|partial|insufficient",
            "data_quality": "good|fair|poor",
            "reasoning": "explanation of your assessment"
        }}
        """)
        
        response = self.llm.invoke([
            HumanMessage(content=analysis_prompt.format(
                query=query,
                result=result
            ))
        ])
        
        try:
            analysis = json.loads(response.content.strip())
            return {
                "confidence": ConfidenceLevel(analysis.get('confidence', 'medium')),
                "completeness": analysis.get('completeness', 'partial'),
                "data": analysis
            }
        except:
            return {
                "confidence": ConfidenceLevel.MEDIUM,
                "completeness": "partial",
                "data": {"reasoning": "Analysis failed"}
            }
    
    def _should_finalize(self, observation: Dict[str, Any], query: str) -> bool:
        """Determine if we have enough information to provide final answer"""
        
        # Check if we have high confidence and complete information
        if (observation['confidence'] == ConfidenceLevel.HIGH and 
            observation.get('completeness') == 'complete'):
            return True
        
        # Check if we've gathered information from multiple sources
        sources_used = set()
        for step in self.reasoning_steps:
            if step.action.startswith('query_'):
                sources_used.add(step.action.split('_')[1])
        
        # If we have medium confidence but multiple sources, we can finalize
        if (observation['confidence'] == ConfidenceLevel.MEDIUM and 
            len(sources_used) >= 2):
            return True
        
        return False
    
    def _finalize_response(self, query: str) -> AgentResponse:
        """Generate final response with confidence and reasoning"""
        
        # Compile all observations
        all_data = []
        sources_used = []
        
        for step in self.reasoning_steps:
            if step.observation and step.observation != "Ready to finalize response":
                all_data.append(step.observation)
                if step.action.startswith('query_'):
                    source = step.action.split('_')[1]
                    if source not in sources_used:
                        sources_used.append(source)
        
        # Generate final synthesized response
        synthesis_prompt = PromptTemplate.from_template("""
        Synthesize a final response based on all the data gathered.
        
        ORIGINAL QUERY: {query}
        
        DATA GATHERED:
        {data}
        
        REASONING STEPS:
        {reasoning}
        
        Provide a comprehensive answer that:
        1. Directly answers the user's query
        2. Synthesizes information from multiple sources if applicable
        3. Includes specific numbers, calculations, and insights
        4. Mentions your confidence level and any limitations
        5. Explains your reasoning process briefly
        
        Format your response clearly with specific data points and analysis.
        """)
        
        reasoning_summary = "\n".join([
            f"Step {i+1}: {step.thought} → {step.action} → {step.observation[:100]}..."
            for i, step in enumerate(self.reasoning_steps)
        ])
        
        response = self.llm.invoke([
            HumanMessage(content=synthesis_prompt.format(
                query=query,
                data="\n\n".join(all_data),
                reasoning=reasoning_summary
            ))
        ])
        
        # Determine overall confidence
        confidence_scores = [step.confidence for step in self.reasoning_steps]
        if all(c == ConfidenceLevel.HIGH for c in confidence_scores):
            overall_confidence = ConfidenceLevel.HIGH
        elif any(c == ConfidenceLevel.HIGH for c in confidence_scores):
            overall_confidence = ConfidenceLevel.MEDIUM
        else:
            overall_confidence = ConfidenceLevel.LOW
        
        return AgentResponse(
            content=response.content.strip(),
            confidence=overall_confidence,
            sources_used=sources_used,
            reasoning_steps=self.reasoning_steps
        )
    
    def _build_context(self, query: str, iteration: int) -> str:
        """Build context from previous reasoning steps"""
        
        if not self.reasoning_steps:
            return "This is the first reasoning step."
        
        context = f"Previous reasoning steps:\n"
        for i, step in enumerate(self.reasoning_steps):
            context += f"Step {i+1}: {step.thought}\n"
            context += f"Action: {step.action}\n"
            context += f"Result: {step.observation[:200]}...\n\n"
        
        return context