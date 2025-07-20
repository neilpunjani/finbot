#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI REST API for Finaptive AI Chatbot
"""
import sys
import os
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

@app.on_event("startup")
async def startup_event():
    """Initialize the pure agentic workflow on startup"""
    global workflow
    try:
        workflow = PureAgenticWorkflow()
        print("✅ Pure Agentic AI workflow initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize pure agentic workflow: {str(e)}")
        raise

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
    if workflow is None:
        raise HTTPException(status_code=503, detail="Pure agentic workflow not initialized")
    
    try:
        status_info = workflow.get_system_status()
        return SystemStatus(
            status="active",
            available_sources=["Adaptive Discovery", "Query Complexity Analysis", "ReAct Cross-Checking", "Multi-Sheet Validation", "Calculation Transparency", "Intelligent Selection"],
            system_info=status_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process a chat message and return response"""
    if workflow is None:
        raise HTTPException(status_code=503, detail="Pure agentic workflow not initialized")
    
    if not message.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Handle special commands
        if message.message.lower() in ['help', 'examples']:
            response = workflow.get_available_commands()
        elif message.message.lower() in ['status', 'system']:
            response = workflow.get_system_status()
        else:
            response = workflow.process_query(message.message)
        
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