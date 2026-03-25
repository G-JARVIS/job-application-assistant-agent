
import streamlit as st
import os
import re
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from pypdf import PdfReader
from io import BytesIO

# --- 1. Configuration & State Definition ---
st.set_page_config(page_title="AlignAgent AI", page_icon="ðŸ¤–", layout="wide")

# Ensure keys are set in Streamlit secrets or environment
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=st.secrets["GEMINI_API_KEY"]
)

class AgentState(TypedDict):
    resume_text: str
    jd_text: str
    company_info: str
    research_analysis: str
    math_result: str
    final_report: str

# --- 2. Tool Definitions ---
def search_company(query: str):
    search = TavilySearchResults(max_results=2)
    return search.run(query)

def calculate_match_score(resume_points: int, total_points: int):
    repl = PythonREPL()
    result = repl.run(f"print(({resume_points} / {total_points}) * 100)")
    return f"{result.strip()}%"

# --- 3. Agent Nodes ---

def researcher_agent(state: AgentState) -> AgentState:
    """Task: Extracts data, researches the company, and identifies gaps."""
    st.write("ðŸ” **Researcher Agent** is analyzing documents and searching the web...")
    
    # Extract Company Name from JD for Search
    company_search_query = f"Company culture and technology stack for the role described in: {state['jd_text'][:500]}"
    web_data = search_company(company_search_query)
    
    prompt = f"""
    Compare the Resume and Job Description below. 
    Use the provided Web Research to contextualize your analysis.
    
    RESUME: {state['resume_text']}
    JD: {state['jd_text']}
    WEB RESEARCH: {web_data}
    
    Output a 'Research Summary' including:
    1. Top 5 required skills.
    2. Company culture alignment.
    3. Missing critical keywords.
    """
    res = llm.invoke(prompt)
    state["research_analysis"] = res.content
    state["company_info"] = web_data
    return state

def advisor_agent(state: AgentState) -> AgentState:
    """Task: Calculates precise match and generates the Markdown report."""
    st.write("ðŸ’¡ **Advisor Agent** is calculating scores and drafting suggestions...")
    
    # Logic to determine points (Simplified for LLM reasoning)
    points_prompt = f"Based on this research: {state['research_analysis']}, count how many of the 10 core JD requirements the candidate meets. Return only the integer."
    matched_count = int(re.findall(r'\d+', llm.invoke(points_prompt).content)[0])
    
    # Use the Calculator Tool
    score = calculate_match_score(matched_count, 10)
    state["math_result"] = score
    
    report_prompt = f"""
    Create a professional Markdown report based on:
    Match Score: {score}
    Research: {state['research_analysis']}
    
    Structure:
    ## ðŸ“Š Match Analysis: {score}
    ## ðŸ¢ Company Insights
    ## ðŸ› ï¸ Recommended Changes
    ## ðŸš€ Project Roadmap
    """
    state["final_report"] = llm.invoke(report_prompt).content
    return state

# --- 4. Streamlit UI Logic ---

def main():
    st.title("ðŸ§­ AlignAgent: Multi-Agent Resume Optimizer")
    
    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    with col2:
        jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")

    if st.button("Run Multi-Agent Analysis", type="primary"):
        if resume_file and jd_file:
            # 1. Extraction
            res_text = "\n".join([p.extract_text() for p in PdfReader(resume_file).pages])
            jd_text = "\n".join([p.extract_text() for p in PdfReader(jd_file).pages])
            
            # 2. Initial State
            state: AgentState = {
                "resume_text": res_text,
                "jd_text": jd_text,
                "company_info": "",
                "research_analysis": "",
                "math_result": "",
                "final_report": ""
            }
            
            # 3. Execution Pipeline (Sequential Agent Handoff)
            with st.status("Agents working...", expanded=True) as status:
                state = researcher_agent(state)
                state = advisor_agent(state)
                status.update(label="Analysis Complete!", state="complete")
            
            # 4. Final Output Display
            st.divider()
            st.markdown(state["final_report"])
            
            st.download_button("Download Report", state["final_report"], "report.md")
        else:
            st.error("Please upload both files.")

if __name__ == "__main__":
    main()
