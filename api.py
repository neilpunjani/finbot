#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI REST API for Finaptive AI Chatbot
"""
import sys
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        os.system("chcp 65001 > nul")
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

from dotenv import load_dotenv, find_dotenv
from src.agents.pure_workflow import PureAgenticWorkflow

# Load environment variables
load_dotenv(find_dotenv(), override=True, verbose=False)

app = FastAPI(
    title="Finaptive Adaptive ReAct Agent with Full Dataset Loading API", 
    description="Fixed: Full Dataset Loading + Data Quality Detection + ReAct Cross-Checking",
    version="8.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global workflow instance
workflow = None
workflow_initializing = False

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    sources: Optional[list] = None

class SystemStatus(BaseModel):
    status: str
    available_sources: list
    system_info: str

async def initialize_workflow():
    """Initialize the workflow lazily when first needed"""
    global workflow, workflow_initializing
    
    if workflow is not None:
        return workflow
    
    if workflow_initializing:
        # Another request is already initializing, wait for it
        while workflow_initializing and workflow is None:
            await asyncio.sleep(0.1)
        return workflow
    
    try:
        workflow_initializing = True
        print("🚀 Initializing Pure Agentic AI workflow...")
        workflow = PureAgenticWorkflow()
        print("✅ Pure Agentic AI workflow initialized successfully!")
        return workflow
    except Exception as e:
        print(f"❌ Failed to initialize pure agentic workflow: {str(e)}")
        raise
    finally:
        workflow_initializing = False

@app.on_event("startup")
async def startup_event():
    """API startup - don't initialize workflow yet"""
    print("🌟 Finaptive AI API started - workflow will initialize on first request")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Finaptive Adaptive ReAct Agent with Full Dataset Loading API",
        "version": "8.1.0", 
        "docs": "/docs",
        "status": "/status",
        "features": ["full_dataset_loading", "data_quality_detection", "adaptive_discovery", "react_cross_checking", "calculation_transparency", "blank_data_penalties", "cached_performance"]
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status and available data sources"""
    global workflow_initializing
    
    # If workflow doesn't exist and isn't initializing, start initialization
    if workflow is None and not workflow_initializing:
        # Don't await - let it initialize in background
        asyncio.create_task(initialize_workflow())
    
    # If still initializing, return loading status
    if workflow_initializing or (workflow and hasattr(workflow, 'is_loading') and workflow.is_loading):
        loading_message = "Initializing system..."
        if workflow and hasattr(workflow, 'loading_status'):
            loading_message = workflow.loading_status
        
        return SystemStatus(
            status="loading",
            available_sources=[],
            system_info=f"System Loading: {loading_message}"
        )
    
    # If workflow is None and not initializing, it failed
    if workflow is None:
        raise HTTPException(status_code=503, detail="Workflow initialization failed")
    
    try:
        # System is ready
        status_info = workflow.get_system_status()
        return SystemStatus(
            status="active",
            available_sources=["RAG Discovery", "Excel Analysis", "CSV Analysis", "Financial Calculations", "Mining Operations", "HR Analytics"],
            system_info=status_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process a chat message and return response"""
    # Ensure workflow is initialized
    current_workflow = await initialize_workflow()
    
    if not message.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Handle special commands
        if message.message.lower() in ['help', 'examples']:
            response = current_workflow.get_available_commands()
        elif message.message.lower() in ['status', 'system']:
            response = current_workflow.get_system_status()
        else:
            response = current_workflow.process_query(message.message)
        
        return ChatResponse(
            response=response,
            session_id=message.session_id,
            sources=[]  # TODO: Extract sources from workflow response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "workflow_ready": workflow is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)