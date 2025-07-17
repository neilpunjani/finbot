import os
import json
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from datetime import datetime

def serialize_for_json(obj):
    """Convert dataclass objects to JSON-serializable format"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        # Handle dataclass objects
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif hasattr(value, '__dict__'):
                result[key] = serialize_for_json(value)
            else:
                result[key] = value
        return result
    else:
        return obj

class AgentState(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    FINALIZING = "finalizing"
    COMPLETE = "complete"

@dataclass
class Thought:
    content: str
    reasoning: str
    confidence: float
    timestamp: datetime

@dataclass
class Action:
    type: str
    parameters: Dict[str, Any]
    reasoning: str
    expected_outcome: str

@dataclass
class Observation:
    result: Any
    success: bool
    insights: List[str]
    confidence: float
    issues: List[str]

@dataclass
class Memory:
    query: str
    thoughts: List[Thought]
    actions: List[Action]
    observations: List[Observation]
    context: Dict[str, Any]

class PureAgent:
    """
    A pure agentic AI system that reasons, plans, acts, and reflects.
    No keywords, no hardcoded rules - just intelligent reasoning.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Agent state
        self.state = AgentState.PLANNING
        self.memory = Memory(
            query="",
            thoughts=[],
            actions=[],
            observations=[],
            context={}
        )
        
        # Tool discovery
        self.available_tools = self._discover_tools()
        
        # Agent personality and capabilities
        self.system_prompt = """
        You are an intelligent data analysis agent with the following capabilities:
        
        CORE ABILITIES:
        - Reason through complex problems step by step
        - Discover and understand data structures dynamically
        - Plan multi-step approaches to achieve goals
        - Reflect on results and adjust strategies
        - Learn from observations and improve performance
        
        REASONING PROCESS:
        1. PLAN: Break down complex queries into logical steps
        2. ACT: Execute actions to gather information or perform analysis
        3. OBSERVE: Analyze results and extract insights
        4. REFLECT: Evaluate success and determine next steps
        5. ADAPT: Modify approach based on what you've learned
        
        PRINCIPLES:
        - Always reason from first principles
        - Discover data structure through exploration, not assumptions
        - Validate findings through multiple approaches when possible
        - Be transparent about confidence levels and limitations
        - Learn from each interaction to improve future performance
        
        You have access to various data sources and tools. Your job is to intelligently 
        figure out how to use them to answer complex questions.
        """
        
    def _discover_tools(self) -> Dict[str, Any]:
        """Dynamically discover available tools and their capabilities"""
        tools = {}
        
        # CSV Data Discovery
        csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            tools['csv_explorer'] = {
                'description': 'Explore and analyze CSV files containing operational data',
                'capabilities': ['data_discovery', 'statistical_analysis', 'pattern_recognition'],
                'files_available': csv_files
            }
        
        # Excel Data Discovery
        excel_path = os.getenv("EXCEL_FILE_PATH")
        if excel_path and os.path.exists(excel_path):
            tools['excel_explorer'] = {
                'description': 'Explore and analyze Excel workbooks with financial data',
                'capabilities': ['worksheet_analysis', 'financial_calculations', 'cross_sheet_analysis'],
                'file_path': excel_path
            }
        
        # Database Discovery
        if os.getenv("DATABASE_URL"):
            tools['sql_explorer'] = {
                'description': 'Query structured database with relational data',
                'capabilities': ['complex_queries', 'joins', 'aggregations'],
                'connection_available': True
            }
        
        # Email Discovery
        if os.getenv("EMAIL_ADDRESS"):
            tools['email_explorer'] = {
                'description': 'Analyze email communications and correspondence',
                'capabilities': ['message_analysis', 'pattern_detection', 'communication_insights'],
                'connection_available': True
            }
        
        return tools
    
    def solve(self, query: str) -> str:
        """Main agent reasoning loop - pure agentic problem solving"""
        
        # Initialize memory for this query
        self.memory = Memory(
            query=query,
            thoughts=[],
            actions=[],
            observations=[],
            context={}
        )
        
        # Start reasoning loop
        max_iterations = 10
        iteration = 0
        
        while self.state != AgentState.COMPLETE and iteration < max_iterations:
            iteration += 1
            
            if self.state == AgentState.PLANNING:
                self._plan_approach()
                # Force transition to execution
                if self.state == AgentState.PLANNING:
                    self.state = AgentState.EXECUTING
            elif self.state == AgentState.EXECUTING:
                self._execute_action()
            elif self.state == AgentState.REFLECTING:
                self._reflect_on_results()
            elif self.state == AgentState.FINALIZING:
                break
                
        # Generate final response
        return self._generate_final_response()
    
    def _plan_approach(self):
        """PLANNING: Analyze query and create execution plan"""
        
        planning_prompt = PromptTemplate.from_template("""
        You are planning how to solve this query using pure reasoning.
        
        QUERY: {query}
        
        AVAILABLE TOOLS:
        {tools}
        
        CONTEXT: {context}
        
        Think step by step:
        1. What is the user really asking for?
        2. What type of data or analysis is needed?
        3. Which tools might be relevant and why?
        4. What's the logical sequence of steps?
        5. What are potential challenges or edge cases?
        
        Create a detailed plan with specific steps. Be intelligent about tool selection
        based on what makes sense for the query, not on keyword matching.
        
        Respond with JSON:
        {{
            "understanding": "what the user is asking for",
            "data_needed": "what type of data is required",
            "approach": "high-level strategy",
            "steps": ["step 1", "step 2", "step 3"],
            "tool_selection_reasoning": "why these tools make sense",
            "potential_challenges": ["challenge 1", "challenge 2"],
            "success_criteria": "how to know if successful"
        }}
        """)
        
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=planning_prompt.format(
                query=self.memory.query,
                tools=json.dumps(self.available_tools, indent=2),
                context=json.dumps(self.memory.context, indent=2)
            ))
        ])
        
        try:
            # Clean the response content
            response_content = response.content.strip()
            
            # Try to extract JSON from the response
            if '```json' in response_content:
                json_start = response_content.find('```json') + 7
                json_end = response_content.find('```', json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
            elif '```' in response_content:
                json_start = response_content.find('```') + 3
                json_end = response_content.find('```', json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
            
            plan = json.loads(response_content)
            
            # Ensure we have valid steps
            steps = plan.get('steps', [])
            if not steps:
                print("⚠️ Warning: No steps found in plan, creating fallback step")
                steps = [f"Analyze available data to answer: {self.memory.query}"]
                plan['steps'] = steps
            
            print(f"🔍 Debug - Plan created with {len(steps)} steps")
            for i, step in enumerate(steps):
                print(f"   Step {i+1}: {step}")
            
            # Store planning results
            self.memory.thoughts.append(Thought(
                content=plan.get('understanding', ''),
                reasoning=plan.get('approach', ''),
                confidence=0.8,
                timestamp=datetime.now()
            ))
            
            self.memory.context.update({
                'plan': plan,
                'current_step': 0,
                'total_steps': len(steps)
            })
            
            # Don't change state here - let the main loop handle it
            
        except Exception as e:
            print(f"❌ Planning failed: {str(e)}")
            print(f"🔍 Debug - Raw response: {response.content[:500]}...")
            self.memory.context['planning_error'] = str(e)
            # Create a fallback plan
            fallback_plan = {
                'understanding': self.memory.query,
                'approach': 'Direct data analysis',
                'steps': [f"Analyze available data to answer: {self.memory.query}"]
            }
            self.memory.context.update({
                'plan': fallback_plan,
                'current_step': 0,
                'total_steps': 1
            })
            self.state = AgentState.EXECUTING  # Continue anyway
    
    def _execute_action(self):
        """EXECUTING: Take intelligent action based on reasoning"""
        
        plan = self.memory.context.get('plan', {})
        current_step = self.memory.context.get('current_step', 0)
        steps = plan.get('steps', [])
        
        # Debug logging
        print(f"🔍 Debug - Current step: {current_step}, Total steps: {len(steps)}")
        if steps:
            print(f"🔍 Debug - Available steps: {steps}")
        
        if current_step >= len(steps):
            print(f"🔍 Debug - No more steps to execute, moving to reflection")
            self.state = AgentState.REFLECTING
            return
        
        current_step_description = steps[current_step]
        print(f"🔍 Debug - Executing step: {current_step_description}")
        
        # Reason about what action to take
        action_prompt = PromptTemplate.from_template("""
        You need to execute this step of your plan through intelligent reasoning.
        
        CURRENT STEP: {step_description}
        OVERALL PLAN: {plan}
        PREVIOUS ACTIONS: {previous_actions}
        PREVIOUS OBSERVATIONS: {previous_observations}
        
        AVAILABLE TOOLS:
        {tools}
        
        Based on pure reasoning, decide what action to take:
        1. Which tool should you use and why?
        2. What specific operation should you perform?
        3. What parameters are needed?
        4. What do you expect to learn?
        
        Be intelligent about tool selection based on the nature of the task,
        not on keyword matching.
        
        Respond with JSON:
        {{
            "action_type": "tool_name",
            "reasoning": "why this tool and action make sense",
            "parameters": {{"key": "value"}},
            "expected_outcome": "what you expect to learn or achieve",
            "confidence": 0.0-1.0
        }}
        """)
        
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=action_prompt.format(
                step_description=current_step_description,
                plan=json.dumps(plan, indent=2),
                previous_actions=json.dumps([serialize_for_json(a) for a in self.memory.actions], indent=2),
                previous_observations=json.dumps([serialize_for_json(o) for o in self.memory.observations], indent=2),
                tools=json.dumps(self.available_tools, indent=2)
            ))
        ])
        
        try:
            # Clean the response content
            response_content = response.content.strip()
            
            # Try to extract JSON from the response
            if '```json' in response_content:
                json_start = response_content.find('```json') + 7
                json_end = response_content.find('```', json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
            elif '```' in response_content:
                json_start = response_content.find('```') + 3
                json_end = response_content.find('```', json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
            
            action_plan = json.loads(response_content)
            
            # Create action
            action = Action(
                type=action_plan.get('action_type', 'explore'),
                parameters=action_plan.get('parameters', {}),
                reasoning=action_plan.get('reasoning', ''),
                expected_outcome=action_plan.get('expected_outcome', '')
            )
            
            print(f"🔍 Debug - Action created: {action.type} with parameters: {action.parameters}")
            
            # Execute the action
            observation = self._execute_tool_action(action)
            
            print(f"🔍 Debug - Action result: success={observation.success}, result={observation.result[:100]}...")
            
            # Store results
            self.memory.actions.append(action)
            self.memory.observations.append(observation)
            
            # Update context
            self.memory.context['current_step'] = current_step + 1
            
            # Decide next state
            if observation.success and current_step + 1 >= len(steps):
                print("🔍 Debug - All steps completed successfully, moving to reflection")
                self.state = AgentState.REFLECTING
            elif not observation.success:
                print("🔍 Debug - Action failed, moving to reflection")
                self.state = AgentState.REFLECTING  # Reflect on failure
            else:
                print(f"🔍 Debug - Step {current_step + 1} completed, continuing execution")
                self.state = AgentState.EXECUTING  # Continue to next step
                
        except Exception as e:
            print(f"❌ Action execution failed: {str(e)}")
            print(f"🔍 Debug - Raw response: {response.content[:500]}...")
            self.memory.observations.append(Observation(
                result=f"Action planning failed: {str(e)}",
                success=False,
                insights=[],
                confidence=0.0,
                issues=[str(e)]
            ))
            self.state = AgentState.REFLECTING
    
    def _execute_tool_action(self, action: Action) -> Observation:
        """Execute a specific tool action with intelligent reasoning"""
        
        tool_type = action.type
        
        if tool_type == 'csv_explorer':
            return self._explore_csv_data(action.parameters)
        elif tool_type == 'excel_explorer':
            return self._explore_excel_data(action.parameters)
        elif tool_type == 'sql_explorer':
            return self._explore_sql_data(action.parameters)
        elif tool_type == 'email_explorer':
            return self._explore_email_data(action.parameters)
        else:
            return Observation(
                result=f"Unknown tool type: {tool_type}",
                success=False,
                insights=[],
                confidence=0.0,
                issues=[f"Tool {tool_type} not recognized"]
            )
    
    def _explore_csv_data(self, parameters: Dict[str, Any]) -> Observation:
        """Intelligently explore CSV data through reasoning"""
        
        try:
            # Discover available CSV files
            csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            
            if not csv_files:
                return Observation(
                    result="No CSV files found",
                    success=False,
                    insights=[],
                    confidence=0.0,
                    issues=["No CSV files available"]
                )
            
            # Reason about which CSV to explore
            exploration_prompt = PromptTemplate.from_template("""
            You need to intelligently explore CSV data to help answer this query.
            
            QUERY: {query}
            ACTION PARAMETERS: {parameters}
            AVAILABLE CSV FILES: {csv_files}
            
            Think about:
            1. Which CSV file is most likely to contain relevant data?
            2. What should you explore first?
            3. What specific analysis would be helpful?
            
            Start by examining the structure and content of the most promising file.
            
            Respond with JSON:
            {{
                "file_to_explore": "filename.csv",
                "reasoning": "why this file makes sense",
                "exploration_type": "structure|sample|analysis",
                "specific_focus": "what to look for"
            }}
            """)
            
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=exploration_prompt.format(
                    query=self.memory.query,
                    parameters=json.dumps(parameters, indent=2),
                    csv_files=csv_files
                ))
            ])
            
            exploration_plan = json.loads(response.content.strip())
            
            # Load and examine the selected CSV
            selected_file = exploration_plan.get('file_to_explore', csv_files[0])
            file_path = os.path.join(csv_dir, selected_file)
            
            df = pd.read_csv(file_path)
            
            # Intelligent data analysis
            analysis_result = self._analyze_dataframe_intelligently(df, selected_file, exploration_plan)
            
            return Observation(
                result=analysis_result,
                success=True,
                insights=[f"Explored {selected_file}", f"Found {len(df)} records", f"Columns: {list(df.columns)}"],
                confidence=0.8,
                issues=[]
            )
            
        except Exception as e:
            return Observation(
                result=f"CSV exploration failed: {str(e)}",
                success=False,
                insights=[],
                confidence=0.0,
                issues=[str(e)]
            )
    
    def _analyze_dataframe_intelligently(self, df: pd.DataFrame, filename: str, exploration_plan: Dict) -> str:
        """Analyze dataframe using pure reasoning"""
        
        # Get basic info
        basic_info = {
            'filename': filename,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'sample_data': df.head(3).to_dict('records')
        }
        
        # Let LLM reason about the data
        analysis_prompt = PromptTemplate.from_template("""
        Analyze this dataframe intelligently to help answer the user's query.
        
        ORIGINAL QUERY: {query}
        EXPLORATION PLAN: {exploration_plan}
        
        DATAFRAME INFO:
        {basic_info}
        
        Using pure reasoning:
        1. What does this data represent?
        2. What insights can you extract relevant to the query?
        3. Are there any patterns or trends?
        4. What calculations or analysis would be helpful?
        5. What additional exploration is needed?
        
        Provide specific insights and analysis, not just descriptions.
        """)
        
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=analysis_prompt.format(
                query=self.memory.query,
                exploration_plan=json.dumps(exploration_plan, indent=2),
                basic_info=json.dumps(basic_info, indent=2)
            ))
        ])
        
        return response.content.strip()
    
    def _explore_excel_data(self, parameters: Dict[str, Any]) -> Observation:
        """Intelligently explore Excel data through reasoning"""
        
        try:
            excel_path = os.getenv("EXCEL_FILE_PATH")
            if not excel_path or not os.path.exists(excel_path):
                return Observation(
                    result="Excel file not found",
                    success=False,
                    insights=[],
                    confidence=0.0,
                    issues=["Excel file not available"]
                )
            
            # Discover worksheet structure
            xls = pd.ExcelFile(excel_path)
            worksheets = xls.sheet_names
            
            # Reason about which worksheet to explore
            worksheet_prompt = PromptTemplate.from_template("""
            You need to intelligently explore Excel data to help answer this query.
            
            QUERY: {query}
            ACTION PARAMETERS: {parameters}
            AVAILABLE WORKSHEETS: {worksheets}
            
            Think about:
            1. Which worksheet is most likely to contain relevant data?
            2. What type of analysis would be helpful?
            3. What should you look for first?
            
            Choose the most promising worksheet and exploration approach.
            
            Respond with JSON:
            {{
                "worksheet_to_explore": "sheet_name",
                "reasoning": "why this worksheet makes sense",
                "exploration_focus": "what to analyze"
            }}
            """)
            
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=worksheet_prompt.format(
                    query=self.memory.query,
                    parameters=json.dumps(parameters, indent=2),
                    worksheets=worksheets
                ))
            ])
            
            worksheet_plan = json.loads(response.content.strip())
            
            # Load and analyze the selected worksheet
            selected_sheet = worksheet_plan.get('worksheet_to_explore', worksheets[0])
            df = pd.read_excel(excel_path, sheet_name=selected_sheet)
            
            # Intelligent analysis
            analysis_result = self._analyze_dataframe_intelligently(df, selected_sheet, worksheet_plan)
            
            return Observation(
                result=analysis_result,
                success=True,
                insights=[f"Explored {selected_sheet} worksheet", f"Found {len(df)} records"],
                confidence=0.8,
                issues=[]
            )
            
        except Exception as e:
            return Observation(
                result=f"Excel exploration failed: {str(e)}",
                success=False,
                insights=[],
                confidence=0.0,
                issues=[str(e)]
            )
    
    def _explore_sql_data(self, parameters: Dict[str, Any]) -> Observation:
        """Intelligently explore SQL data through reasoning"""
        # This would use the existing SQL agent but with intelligent reasoning
        return Observation(
            result="SQL exploration not implemented yet",
            success=False,
            insights=[],
            confidence=0.0,
            issues=["SQL exploration needs implementation"]
        )
    
    def _explore_email_data(self, parameters: Dict[str, Any]) -> Observation:
        """Intelligently explore email data through reasoning"""
        # This would use the existing email agent but with intelligent reasoning
        return Observation(
            result="Email exploration not implemented yet",
            success=False,
            insights=[],
            confidence=0.0,
            issues=["Email exploration needs implementation"]
        )
    
    def _reflect_on_results(self):
        """REFLECTING: Analyze results and determine next steps"""
        
        reflection_prompt = PromptTemplate.from_template("""
        Reflect on your progress and determine what to do next.
        
        ORIGINAL QUERY: {query}
        PLAN: {plan}
        ACTIONS TAKEN: {actions}
        OBSERVATIONS: {observations}
        
        Analyze:
        1. How successful have you been so far?
        2. What have you learned that's relevant to the query?
        3. What's still missing or unclear?
        4. Do you need to explore more data or can you provide an answer?
        5. If you need more exploration, what should you do next?
        
        Respond with JSON:
        {{
            "progress_assessment": "how well you're doing",
            "key_findings": ["finding 1", "finding 2"],
            "missing_information": ["what's still needed"],
            "next_action": "continue_exploring|finalize_answer|revise_plan",
            "reasoning": "why this next step makes sense",
            "confidence": 0.0-1.0
        }}
        """)
        
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=reflection_prompt.format(
                query=self.memory.query,
                plan=json.dumps(self.memory.context.get('plan', {}), indent=2),
                actions=json.dumps([serialize_for_json(a) for a in self.memory.actions], indent=2),
                observations=json.dumps([serialize_for_json(o) for o in self.memory.observations], indent=2)
            ))
        ])
        
        try:
            # Clean the response content
            response_content = response.content.strip()
            
            # Try to extract JSON from the response
            if '```json' in response_content:
                json_start = response_content.find('```json') + 7
                json_end = response_content.find('```', json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
            elif '```' in response_content:
                json_start = response_content.find('```') + 3
                json_end = response_content.find('```', json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
            
            reflection = json.loads(response_content)
            
            # Store reflection
            self.memory.thoughts.append(Thought(
                content=reflection.get('progress_assessment', ''),
                reasoning=reflection.get('reasoning', ''),
                confidence=reflection.get('confidence', 0.5),
                timestamp=datetime.now()
            ))
            
            # Determine next state
            next_action = reflection.get('next_action', 'finalize_answer')
            
            print(f"🔍 Debug - Reflection completed, next action: {next_action}")
            
            if next_action == 'continue_exploring':
                print("🔍 Debug - Continuing exploration")
                self.state = AgentState.EXECUTING
            elif next_action == 'revise_plan':
                print("🔍 Debug - Revising plan")
                self.state = AgentState.PLANNING
            else:
                print("🔍 Debug - Finalizing answer")
                self.state = AgentState.FINALIZING
                
        except Exception as e:
            print(f"❌ Reflection failed: {str(e)}")
            print(f"🔍 Debug - Raw response: {response.content[:500]}...")
            self.state = AgentState.FINALIZING  # If reflection fails, try to finalize
    
    def _generate_final_response(self) -> str:
        """Generate comprehensive final response"""
        
        response_prompt = PromptTemplate.from_template("""
        Generate a comprehensive final response based on your reasoning and findings.
        
        ORIGINAL QUERY: {query}
        THOUGHTS: {thoughts}
        ACTIONS TAKEN: {actions}
        OBSERVATIONS: {observations}
        
        Create a response that:
        1. Directly answers the user's question
        2. Provides specific insights and data
        3. Explains your reasoning process
        4. Indicates confidence level
        5. Suggests follow-up questions or additional analysis if relevant
        
        Be specific, insightful, and transparent about your reasoning process.
        """)
        
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=response_prompt.format(
                query=self.memory.query,
                thoughts=json.dumps([serialize_for_json(t) for t in self.memory.thoughts], indent=2),
                actions=json.dumps([serialize_for_json(a) for a in self.memory.actions], indent=2),
                observations=json.dumps([serialize_for_json(o) for o in self.memory.observations], indent=2)
            ))
        ])
        
        final_response = response.content.strip()
        
        # Add agent transparency
        final_response += "\n\n" + "="*50 + "\n"
        final_response += "🤖 **Agent Reasoning Process:**\n"
        final_response += f"• Thoughts: {len(self.memory.thoughts)}\n"
        final_response += f"• Actions taken: {len(self.memory.actions)}\n"
        final_response += f"• Observations made: {len(self.memory.observations)}\n"
        
        # Calculate overall confidence
        if self.memory.observations:
            avg_confidence = sum(o.confidence for o in self.memory.observations) / len(self.memory.observations)
            confidence_text = "High" if avg_confidence >= 0.7 else "Medium" if avg_confidence >= 0.5 else "Low"
            final_response += f"• Overall confidence: {confidence_text} ({avg_confidence:.1f})\n"
        
        return final_response