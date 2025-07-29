import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dataclasses import dataclass
import pickle
from pathlib import Path
import hashlib

@dataclass
class DataSourceSchema:
    """Schema information for a data source"""
    source_name: str
    file_path: str
    sheet_name: Optional[str] = None
    columns: List[str] = None
    sample_values: Dict[str, List] = None
    data_types: Dict[str, str] = None
    row_count: int = 0
    domain_info: Dict[str, Any] = None

@dataclass
class SourceCandidate:
    """Candidate source from RAG search"""
    schema: DataSourceSchema
    relevance_score: float
    reason: str

class RAGDataIndexer:
    """Builds and maintains vector index of all data sources"""
    
    def __init__(self, index_path: str = "data_index"):
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        self.vector_store = None
        self.schema_store = {}
        self.domain_knowledge = self._load_domain_knowledge()
        
        # Create index directory
        Path(index_path).mkdir(exist_ok=True)
        
    def _load_domain_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Pre-built domain knowledge for different sheet types"""
        return {
            # Financial data patterns
            "financial": {
                "keywords": ["revenue", "income", "profit", "assets", "liabilities", "cash", "expense"],
                "structures": ["hierarchical", "level1", "level2", "level3"],
                "expertise": "financial_analyst",
                "calculation_patterns": {
                    "cash_ratio": ["cash", "current_liabilities"],
                    "current_ratio": ["current_assets", "current_liabilities"],
                    "debt_ratio": ["total_debt", "total_assets"],
                    "net_income": ["revenue", "expenses"]
                }
            },
            # Mining operations patterns
            "mining": {
                "keywords": ["production", "ore", "metal", "grade", "recovery", "tonnes", "mining"],
                "structures": ["operational", "time_series"],
                "expertise": "mining_operations_analyst",
                "calculation_patterns": {
                    "recovery_rate": ["metal_produced", "ore_processed"],
                    "grade": ["metal_content", "ore_volume"],
                    "throughput": ["ore_processed", "time_period"]
                }
            },
            # HR/Training patterns
            "hr": {
                "keywords": ["training", "hours", "employee", "certification", "safety"],
                "structures": ["employee_based", "time_series"],
                "expertise": "hr_analyst",
                "calculation_patterns": {
                    "training_hours": ["employee", "hours", "training_type"],
                    "certification_rate": ["certified", "total_employees"]
                }
            }
        }
    
    def build_index(self) -> None:
        """Build complete index of all data sources"""
        print("🔍 Building RAG index for all data sources...")
        
        documents = []
        schemas = {}
        
        # Index Excel files
        excel_path = os.getenv("EXCEL_FILE_PATH")
        if excel_path and os.path.exists(excel_path):
            excel_docs, excel_schemas = self._index_excel_file(excel_path)
            documents.extend(excel_docs)
            schemas.update(excel_schemas)
        
        # Index CSV files
        csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            for csv_file in csv_files:
                csv_path = os.path.join(csv_dir, csv_file)
                csv_docs, csv_schemas = self._index_csv_file(csv_path)
                documents.extend(csv_docs)
                schemas.update(csv_schemas)
        
        # Create vector store
        if documents:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.index_path
            )
            print(f"✅ Indexed {len(documents)} data sources")
            
            # DEBUG: Show what was actually indexed
            print("🔍 DEBUG: Indexed sources:")
            for i, (key, schema) in enumerate(schemas.items(), 1):
                print(f"   {i}. {schema.source_name} - {len(schema.columns)} columns")
            
            # Save schemas
            self.schema_store = schemas
            self._save_schemas()
        else:
            print("❌ No data sources found to index")
    
    def _index_excel_file(self, excel_path: str) -> Tuple[List[Document], Dict[str, DataSourceSchema]]:
        """Index all sheets in an Excel file"""
        documents = []
        schemas = {}
        
        try:
            xls = pd.ExcelFile(excel_path)
            print(f"   📊 Indexing Excel file: {excel_path}")
            print(f"   📋 Found {len(xls.sheet_names)} sheets: {xls.sheet_names}")
            
            for sheet_name in xls.sheet_names:
                try:
                    # Just fucking read the sheet - no filtering
                    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
                    
                    # Only skip if completely empty
                    if df.empty:
                        print(f"     ⚠️ Skipping {sheet_name}: Completely empty")
                        continue
                    
                    # Create schema with clean sheet name
                    clean_sheet_name = sheet_name.strip()
                    schema = self._create_schema(excel_path, clean_sheet_name, df)
                    schema_key = f"excel_{clean_sheet_name}_{hash(excel_path + clean_sheet_name) % 10000}"  # Add hash to prevent collisions
                    schemas[schema_key] = schema
                    
                    # Create searchable document
                    content = self._create_document_content(schema)
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source_type": "excel",
                            "file_path": excel_path,
                            "sheet_name": clean_sheet_name,
                            "schema_key": schema_key
                        }
                    )
                    documents.append(doc)
                    
                    print(f"     ✅ Indexed sheet: {sheet_name} ({df.shape[0]} rows, {df.shape[1]} cols)")
                    print(f"         Columns: {list(df.columns)}")
                    
                    # DEBUG: Show what content is being indexed for semantic search
                    if sheet_name in ['VW_PBI', 'TB', 'By COA Group']:  # Debug key financial sheets
                        content = self._create_document_content(schema)
                        print(f"         INDEXED CONTENT: {content[:500]}...")  # First 500 chars
                        
                        # Show unique value counts per column
                        total_unique_values = 0
                        for col, values in schema.sample_values.items():
                            if values:
                                total_unique_values += len(values)
                                print(f"         Column '{col}': {len(values)} unique values")
                        print(f"         TOTAL UNIQUE VALUES INDEXED: {total_unique_values}")
                    
                except Exception as e:
                    print(f"     ❌ Failed to index sheet {sheet_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"❌ Failed to index Excel file {excel_path}: {str(e)}")
        
        return documents, schemas
    
    def _index_csv_file(self, csv_path: str) -> Tuple[List[Document], Dict[str, DataSourceSchema]]:
        """Index a CSV file"""
        documents = []
        schemas = {}
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty or df.shape[1] < 2:
                return documents, schemas
            
            file_name = os.path.basename(csv_path).replace('.csv', '')
            
            # Create schema
            schema = self._create_schema(csv_path, None, df)
            schema_key = f"csv_{file_name}"
            schemas[schema_key] = schema
            
            # Create searchable document
            content = self._create_document_content(schema)
            doc = Document(
                page_content=content,
                metadata={
                    "source_type": "csv",
                    "file_path": csv_path,
                    "schema_key": schema_key
                }
            )
            documents.append(doc)
            
            print(f"   📄 Indexed CSV: {file_name} ({df.shape[0]} rows, {df.shape[1]} cols)")
            
        except Exception as e:
            print(f"❌ Failed to index CSV file {csv_path}: {str(e)}")
        
        return documents, schemas
    
    def _create_schema(self, file_path: str, sheet_name: Optional[str], df: pd.DataFrame) -> DataSourceSchema:
        """Create schema information for a data source"""
        
        # Basic schema info
        columns = list(df.columns)
        data_types = {col: str(df[col].dtype) for col in columns}
        
        # Only embed TEXT values, skip numbers
        sample_values = {}
        for col in columns:
            if df[col].dtype == 'object':  # Only text columns
                unique_vals = df[col].dropna().unique()
                
                # Filter out numeric-like values
                text_vals = []
                for val in unique_vals:
                    val_str = str(val)
                    # Skip if it's just numbers or mostly numbers
                    if not val_str.replace('.', '').replace('-', '').replace(' ', '').isdigit():
                        text_vals.append(val)
                
                if text_vals:
                    # Limit to 100 values max to stay within token limits
                    sample_values[col] = text_vals[:100]
        
        # Detect domain type
        domain_info = self._analyze_domain(columns, sample_values, sheet_name)
        
        return DataSourceSchema(
            source_name=sheet_name or os.path.basename(file_path),
            file_path=file_path,
            sheet_name=sheet_name,
            columns=columns,
            sample_values=sample_values,
            data_types=data_types,
            row_count=len(df),
            domain_info=domain_info
        )
    
    def _analyze_domain(self, columns: List[str], sample_values: Dict, sheet_name: Optional[str]) -> Dict[str, Any]:
        """Analyze what domain this data belongs to"""
        
        columns_text = ' '.join(columns).lower()
        sheet_text = (sheet_name or '').lower()
        sample_text = ' '.join([str(v) for vals in sample_values.values() for v in vals]).lower()
        
        all_text = f"{columns_text} {sheet_text} {sample_text}"
        
        # Score each domain
        domain_scores = {}
        for domain, info in self.domain_knowledge.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in all_text:
                    score += 1
            domain_scores[domain] = score
        
        # Get best domain
        best_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "unknown"
        
        return {
            "domain_type": best_domain,
            "domain_confidence": domain_scores.get(best_domain, 0),
            "expertise_needed": self.domain_knowledge.get(best_domain, {}).get("expertise", "general_analyst"),
            "structure_type": self._detect_structure_type(columns),
            "calculation_capabilities": self._detect_calculations(columns, best_domain)
        }
    
    def _detect_structure_type(self, columns: List[str]) -> str:
        """Detect the structure type of the data"""
        columns_lower = [col.lower() for col in columns]
        
        if any('level1' in col or 'level2' in col or 'level3' in col for col in columns_lower):
            return "hierarchical"
        elif any('date' in col or 'time' in col or 'year' in col for col in columns_lower):
            return "time_series"
        elif any('company' in col or 'entity' in col for col in columns_lower):
            return "entity_based"
        else:
            return "tabular"
    
    def _detect_calculations(self, columns: List[str], domain: str) -> List[str]:
        """Detect what calculations are possible with these columns"""
        columns_lower = [col.lower() for col in columns]
        capabilities = []
        
        if domain in self.domain_knowledge:
            patterns = self.domain_knowledge[domain].get("calculation_patterns", {})
            
            for calc_name, required_cols in patterns.items():
                if all(any(req_col in col for col in columns_lower) for req_col in required_cols):
                    capabilities.append(calc_name)
        
        return capabilities
    
    def _create_document_content(self, schema: DataSourceSchema) -> str:
        """Create searchable content from schema - RAW DATA FOR SEMANTIC MATCHING"""
        
        content_parts = [
            f"Sheet: {schema.source_name}",
            f"Columns: {' '.join(schema.columns)}",  # All column names for matching
        ]
        
        # Add ALL sample values for semantic matching - NO FILTERING
        if schema.sample_values:
            for col, values in schema.sample_values.items():
                if values:
                    # Include column name + all values for semantic search
                    values_text = ' '.join(map(str, values))
                    content_parts.append(f"{col}: {values_text}")
        
        return '\n'.join(content_parts)
    
    def _save_schemas(self) -> None:
        """Save schemas to disk"""
        schema_file = os.path.join(self.index_path, "schemas.pkl")
        with open(schema_file, 'wb') as f:
            pickle.dump(self.schema_store, f)
    
    def load_existing_index(self) -> bool:
        """Load existing index if available"""
        try:
            if os.path.exists(self.index_path):
                self.vector_store = Chroma(
                    persist_directory=self.index_path,
                    embedding_function=self.embeddings
                )
                
                # Load schemas
                schema_file = os.path.join(self.index_path, "schemas.pkl")
                if os.path.exists(schema_file):
                    with open(schema_file, 'rb') as f:
                        self.schema_store = pickle.load(f)
                
                print(f"✅ Loaded existing index with {len(self.schema_store)} schemas")
                return True
        except Exception as e:
            print(f"❌ Failed to load existing index: {str(e)}")
        
        return False


class RAGSourceSelector:
    """LLM-based intelligent source selection using schema reasoning"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def select_best_source(self, query: str, candidates: List[SourceCandidate]) -> Optional[SourceCandidate]:
        """Select the best source using LLM reasoning over schemas"""
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Build schema context for LLM
        schema_context = self._build_schema_context(candidates)
        
        prompt = f"""You are a data analyst. Your job is to select the ONE source that can answer this query.

QUERY: {query}

AVAILABLE SOURCES:
{schema_context}

INSTRUCTIONS:
1. Read the query carefully
2. Determine what EXACT data is needed to answer it
3. Check which source has that EXACT data
4. If NO source has the required data, say "NO MATCH"

EXAMPLES:
- Query "cash ratio" needs: Cash + Current Liabilities data
- Query "revenue for Ontario" needs: Revenue data + Ontario geographic data  
- Query "training hours for gold" needs: Training data + Gold/commodity data

BE STRICT:
- If query asks for "cash ratio" but no source has cash/liability data → NO MATCH
- If query asks for "Ontario revenue" but no source has Ontario data → NO MATCH
- If query asks for calculation but source lacks required columns → NO MATCH

RESPONSE FORMAT:
Best source: [SOURCE_NAME] or NO MATCH
Reason: [Why this source has the exact data needed, or why no source matches]

RESPONSE:"""

        try:
            response = self.llm.invoke(prompt).content
            print(f"   🧠 LLM Decision: {response}")
            
            # Check for NO MATCH
            if "NO MATCH" in response.upper():
                print("   ❌ LLM determined no source has the required data")
                return None
            
            selected_candidate = self._parse_selection_response(response, candidates)
            
            if selected_candidate:
                print(f"   ✅ Selected: {selected_candidate.schema.source_name}")
                reasoning = response.split('Reason:')[-1].strip() if 'Reason:' in response else 'No reason provided'
                print(f"   🎯 Reasoning: {reasoning}")
            else:
                print("   ❌ Failed to parse LLM response, no source selected")
            
            return selected_candidate
        except Exception as e:
            print(f"❌ LLM selection failed: {str(e)}")
            return None  # Don't fallback, let it fail properly
    
    
    def _build_schema_context(self, candidates: List[SourceCandidate]) -> str:
        """Build readable schema context for LLM"""
        
        context_lines = []
        
        for i, candidate in enumerate(candidates, 1):
            schema = candidate.schema
            
            context_lines.append(f"\n{i}. SOURCE: {schema.source_name}")
            context_lines.append(f"   File: {os.path.basename(schema.file_path)}")
            context_lines.append(f"   Domain: {schema.domain_info.get('domain_type', 'unknown')}")
            context_lines.append(f"   Structure: {schema.domain_info.get('structure_type', 'tabular')}")
            context_lines.append(f"   Row count: {schema.row_count:,}")
            context_lines.append(f"   COLUMNS: {', '.join(schema.columns)}")
            
            # Highlight financial-relevant columns
            financial_keywords = ['cash', 'liability', 'asset', 'revenue', 'income', 'debt', 'equity']
            relevant_cols = [col for col in schema.columns if any(keyword in col.lower() for keyword in financial_keywords)]
            if relevant_cols:
                context_lines.append(f"   FINANCIAL COLUMNS: {', '.join(relevant_cols)}")
            
            # Add hierarchical info for Level-based data
            if any('level' in col.lower() for col in schema.columns):
                context_lines.append(f"   HIERARCHICAL: Contains Level1/Level2/Level3 structure for line item analysis")
            
            # Show sample values for key columns
            if schema.sample_values:
                important_samples = []
                # Prioritize financial and hierarchical columns
                priority_cols = [col for col in schema.columns if 
                               any(keyword in col.lower() for keyword in ['cash', 'liability', 'asset', 'revenue', 'level1', 'level2'])]
                
                for col in priority_cols[:3]:  # Show top 3 important columns
                    if col in schema.sample_values and schema.sample_values[col]:
                        values = schema.sample_values[col]
                        sample_str = ', '.join(map(str, values[:3]))
                        important_samples.append(f"{col}: [{sample_str}]")
                
                if important_samples:
                    context_lines.append(f"   KEY DATA: {' | '.join(important_samples)}")
            
            context_lines.append(f"   Calculation capabilities: {', '.join(schema.domain_info.get('calculation_capabilities', ['basic analysis']))}")
        
        return '\n'.join(context_lines)
    
    def _parse_selection_response(self, response: str, candidates: List[SourceCandidate]) -> Optional[SourceCandidate]:
        """Parse LLM response to get selected source"""
        
        response_lower = response.lower()
        
        # Try to match source names
        for candidate in candidates:
            source_name_lower = candidate.schema.source_name.lower()
            if source_name_lower in response_lower:
                return candidate
        
        # Fallback: try to extract by number (1., 2., etc.)
        for i, candidate in enumerate(candidates, 1):
            if f"source: {i}" in response_lower or f"best source: {i}" in response_lower:
                return candidate
        
        # Final fallback: highest relevance score
        return max(candidates, key=lambda c: c.relevance_score)


class RAGDiscoveryAgent:
    """Main RAG-based discovery agent that replaces the slow current discovery"""
    
    def __init__(self, rebuild_index: bool = False):
        self.indexer = RAGDataIndexer()
        self.selector = RAGSourceSelector()
        
        # Load or build index
        if not rebuild_index and self.indexer.load_existing_index():
            print("✅ RAG Discovery Agent ready (using existing index)")
        else:
            print("🔄 Building new RAG index...")
            self.indexer.build_index()
            print("✅ RAG Discovery Agent ready (new index built)")
    
    def discover_best_source(self, query: str, top_k: int = 5) -> Optional[SourceCandidate]:
        """Main discovery method - Financial queries → VW_PBI, Others → RAG"""
        
        if not self.indexer.vector_store:
            print("❌ No vector store available")
            return None
        
        print(f"🔍 RAG Discovery: {query}")
        
        # Step 1: Check if this is a financial query
        if self._is_financial_query(query):
            print("   💰 Financial query detected → Selecting VW_PBI")
            return self._get_vw_pbi_source()
        
        # Step 2: For non-financial queries, use RAG
        print("   🔍 Operational query → Using RAG search")
        try:
            docs = self.indexer.vector_store.similarity_search_with_score(query, k=top_k * 3)
            print(f"   📊 Found {len(docs)} potential sources from vector search")
        except Exception as e:
            print(f"❌ Vector search failed: {str(e)}")
            return None
        
        # Step 2: Get top 5 unique sources - SIMPLE
        candidates = []
        seen_source_names = set()
        
        print(f"   📋 Top 5 unique sources:")
        
        for doc, score in docs:
            if len(candidates) >= 5:  # Stop at 5
                break
                
            schema_key = doc.metadata.get('schema_key')
            if schema_key not in self.indexer.schema_store:
                continue
                
            schema = self.indexer.schema_store[schema_key]
            
            # Skip if we already have this source name
            if schema.source_name in seen_source_names:
                continue
                
            seen_source_names.add(schema.source_name)
            
            print(f"      {len(candidates)+1}. {schema.source_name} (similarity: {1.0 - score:.3f})")
            print(f"         Columns: {', '.join(schema.columns[:5])}{'...' if len(schema.columns) > 5 else ''}")
            
            candidate = SourceCandidate(
                schema=schema,
                relevance_score=1.0 - score,
                reason=f"Vector similarity: {1.0 - score:.3f}"
            )
            candidates.append(candidate)
        
        if not candidates:
            print("❌ No valid candidates found")
            return None
        
        # Step 3: LLM-based intelligent selection
        print(f"   🧠 LLM selecting best from {len(candidates)} candidates...")
        best_source = self.selector.select_best_source(query, candidates)
        
        if best_source:
            print(f"   ✅ Selected: {best_source.schema.source_name}")
            print(f"   📋 Expertise needed: {best_source.schema.domain_info.get('expertise_needed', 'analyst')}")
        
        return best_source
    
    def get_schema_info(self, schema_key: str) -> Optional[DataSourceSchema]:
        """Get detailed schema information for a source"""
        return self.indexer.schema_store.get(schema_key)
    
    def rebuild_index(self) -> None:
        """Force rebuild of the entire index"""
        print("🔄 Rebuilding RAG index...")
        self.indexer.build_index()
        print("✅ RAG index rebuilt")
    
    def _is_financial_query(self, query: str) -> bool:
        """Check if query is financial-related using LLM classification"""
        
        classification_prompt = f"""Is this query related to financial data, accounting, or financial analysis?

Query: {query}

Financial queries include:
- Financial ratios (any ratio calculation)
- Revenue, income, profit, loss analysis  
- Balance sheet items (assets, liabilities, equity)
- Cash flow analysis
- Financial performance metrics
- Accounting data analysis
- Budget, cost, expense analysis
- Investment returns, margins

Respond with only: YES or NO

Response:"""

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
            response = llm.invoke(classification_prompt).content.strip().upper()
            
            is_financial = "YES" in response
            print(f"   🤖 LLM Classification: {query} → {'Financial' if is_financial else 'Operational'}")
            return is_financial
            
        except Exception as e:
            print(f"   ⚠️ LLM classification failed: {str(e)}, using fallback")
            # Fallback to basic keyword matching
            financial_terms = ['ratio', 'revenue', 'income', 'profit', 'cash', 'financial', 'balance', 'cost']
            return any(term in query.lower() for term in financial_terms)
    
    def _get_vw_pbi_source(self) -> Optional[SourceCandidate]:
        """Get VW_PBI source directly"""
        # Find VW_PBI in schema store
        for schema_key, schema in self.indexer.schema_store.items():
            if schema.source_name == 'VW_PBI':
                return SourceCandidate(
                    schema=schema,
                    relevance_score=1.0,  # Perfect match for financial queries
                    reason="Financial query → VW_PBI selected automatically"
                )
        
        print("   ❌ VW_PBI not found in indexed sources")
        return None