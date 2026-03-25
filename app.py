
import json
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
import os
import google.generativeai as genai

import streamlit as st
from pypdf import PdfReader


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="Job Application Assistant Agent",
    page_icon="🧭",
    layout="wide",
)

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash-lite")


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you", "are", "was",
    "were", "will", "have", "has", "had", "not", "but", "our", "their", "they", "them",
    "been", "into", "about", "over", "under", "role", "job", "position", "candidate",
    "work", "working", "experience", "skills", "skill", "responsibilities", "responsibility",
    "required", "preferred", "must", "should", "able", "ability", "team", "teams",
    "company", "business", "project", "projects", "resume", "cv", "assistant", "engineer",
    "developer", "analyst", "manager", "associate", "senior", "junior", "intern", "lead",
    "help", "apply", "application", "opportunity", "opportunities", "using", "use",
    "based", "build", "built", "make", "made", "etc", "etc.", "more", "less", "may",
    "can", "could", "would", "should", "shall", "need", "needed", "needs",
}

SKILL_CATALOG = [
    # Core technical
    "python", "java", "javascript", "typescript", "sql", "mysql", "postgresql",
    "mongodb", "html", "css", "react", "node.js", "express", "django", "flask", "fastapi",
    "rest api", "api", "git", "github", "linux", "docker", "kubernetes", "aws", "azure",
    "gcp", "power bi", "tableau", "excel", "pandas", "numpy", "matplotlib", "seaborn",
    "scikit-learn", "tensorflow", "pytorch", "nlp", "machine learning", "deep learning",
    "data analysis", "data visualization", "statistics", "spark", "hadoop", "airflow",
    "streamlit", "langchain", "langgraph", "llm", "prompt engineering", "rag",
    # Software/process
    "object oriented programming", "oop", "data structures", "algorithms", "testing",
    "debugging", "agile", "scrum", "jira", "postman", "ci/cd", "software development",
    "project management", "research", "communication", "teamwork", "leadership",
    "problem solving", "analytical thinking", "presentation",
    # Resume / job-related phrases
    "collaboration", "stakeholder management", "quantitative analysis", "documentation",
    "model deployment", "feature engineering", "etl", "database design",
]

ACTION_VERBS = {
    "developed", "built", "designed", "implemented", "created", "improved", "optimized",
    "analyzed", "managed", "led", "automated", "collaborated", "delivered", "reduced",
    "increased", "decreased", "generated", "planned", "organized", "resolved", "crafted",
    "produced", "integrated", "deployed", "tested", "maintained",
}

SECTION_MARKERS = {
    "summary": ["summary", "profile", "objective", "about me", "professional summary"],
    "experience": ["experience", "work experience", "internship", "employment"],
    "projects": ["projects", "project work", "academic projects", "selected projects"],
    "skills": ["skills", "technical skills", "core skills"],
    "education": ["education", "academics"],
    "certifications": ["certifications", "certificates", "license", "licenses"],
}


# -----------------------------
# Text extraction
# -----------------------------
def extract_text_from_upload(uploaded_file) -> str:
    """Read text from uploaded PDF/TXT file."""
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    suffix = Path(name).suffix

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()

    try:
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        return str(raw)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\-\+\.]{2,}", text.lower())


# -----------------------------
# Skill and keyword extraction
# -----------------------------
def extract_skills(text: str):
    raw = text.lower()
    found = []
    for skill in SKILL_CATALOG:
        pattern = re.escape(skill.lower())
        if re.search(pattern, raw):
            found.append(skill)
    # sort by appearance length and remove duplicates
    found = sorted(set(found), key=lambda s: (len(s), s))
    return found


def extract_top_keywords(text: str, top_n: int = 12):
    tokens = tokenize(text)
    filtered = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) < 3:
            continue
        filtered.append(token)
    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(top_n)]


def detect_sections(text: str):
    raw = normalize_text(text)
    result = {}
    for section, markers in SECTION_MARKERS.items():
        result[section] = any(marker in raw for marker in markers)
    return result


def count_action_verbs(text: str):
    tokens = set(tokenize(text))
    return sorted(tokens.intersection(ACTION_VERBS))


def contains_numbers(text: str) -> bool:
    return bool(re.search(r"\d", text))


# -----------------------------
# Agent-style analysis
# -----------------------------
def build_tailored_summary(matched_skills, jd_keywords, section_flags):
    skill_part = ", ".join(matched_skills[:5]) if matched_skills else "relevant technical and analytical skills"
    keyword_part = ", ".join(jd_keywords[:4]) if jd_keywords else "the target role"

    if section_flags.get("experience"):
        tone = "experienced"
    elif section_flags.get("projects"):
        tone = "hands-on"
    else:
        tone = "motivated"

    return (
        f"{tone.capitalize()} candidate with practical exposure to {skill_part}, "
        f"seeking opportunities involving {keyword_part}. "
        "Focused on delivering measurable results, learning quickly, and contributing to team goals."
    )


def score_resume(jd_skills, resume_skills, jd_keywords, resume_text, section_flags):
    skill_match_count = len(set(jd_skills).intersection(resume_skills))
    skill_coverage = skill_match_count / max(len(jd_skills), 1)

    resume_tokens = set(tokenize(resume_text))
    keyword_match_count = sum(1 for kw in jd_keywords if kw in resume_tokens)
    keyword_coverage = keyword_match_count / max(len(jd_keywords), 1)

    sections_score = (
        (1 if section_flags["summary"] else 0)
        + (1 if section_flags["experience"] else 0)
        + (1 if section_flags["projects"] else 0)
        + (1 if section_flags["skills"] else 0)
        + (1 if section_flags["education"] else 0)
    ) / 5

    verb_count = len(count_action_verbs(resume_text))
    verb_score = min(verb_count, 5) / 5

    overall = round(
        100
        * (
            0.45 * skill_coverage
            + 0.25 * keyword_coverage
            + 0.20 * sections_score
            + 0.10 * verb_score
        )
    )

    return {
        "overall": max(0, min(overall, 100)),
        "skill_coverage": round(skill_coverage * 100, 1),
        "keyword_coverage": round(keyword_coverage * 100, 1),
        "sections_score": round(sections_score * 100, 1),
        "verb_score": round(verb_score * 100, 1),
        "skill_match_count": skill_match_count,
        "skill_total_count": len(jd_skills),
    }


def build_suggestions(resume_text, jd_text, matched_skills, missing_skills, jd_keywords, section_flags):
    suggestions = []

    if not section_flags["summary"]:
        suggestions.append("Add a short professional summary at the top of the resume.")

    if not section_flags["skills"]:
        suggestions.append("Create a dedicated Skills section so ATS tools can detect keywords quickly.")

    if not section_flags["projects"] and not section_flags["experience"]:
        suggestions.append("Add a Projects or Experience section with 2–4 strong bullet points.")

    if missing_skills:
        top_missing = missing_skills[:5]
        suggestions.append(
            "Mention or learn these missing JD skills: " + ", ".join(top_missing) + "."
        )

    if jd_keywords:
        suggestions.append(
            "Mirror some JD keywords naturally in your bullets and summary: "
            + ", ".join(jd_keywords[:5])
            + "."
        )

    if not contains_numbers(resume_text):
        suggestions.append("Add measurable impact with numbers, percentages, or outcomes.")

    verbs = count_action_verbs(resume_text)
    if len(verbs) < 3:
        suggestions.append(
            "Use stronger action verbs such as developed, implemented, improved, or led."
        )

    if len(suggestions) < 4:
        suggestions.append("Keep bullets concise, role-focused, and aligned to the target job.")

    return suggestions

def generate_llm_suggestions(missing_skills, job_role):
    if not missing_skills:
        return ""
    
    prompt = f"""
    You are a professional resume advisor.

    Job Role: {job_role}

    Missing Skills:
    {', '.join(missing_skills)}

    Task:
    1. Suggest how the candidate can improve their resume.
    2. Recommend skills to learn.
    3. Suggest certifications or projects.
    4. Provide professional resume improvement suggestions.

    Output format:
    - Improvement Suggestions
    - Skills to Learn
    - Recommended Projects
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"

def run_agent(resume_text: str, jd_text: str):
    resume_text = (resume_text or "").strip()
    jd_text = (jd_text or "").strip()

    if not resume_text or not jd_text:
        return {"error": "Both resume text and job description text are required."}

    goal = "Compare a resume with a job description and generate a tailored improvement report."

    plan = [
        "Read and clean both documents.",
        "Extract skills, keywords, and resume sections.",
        "Compare matched and missing requirements.",
        "Score the resume against the job description.",
        "Suggest focused improvements and a tailored summary.",
    ]

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    resume_keywords = extract_top_keywords(resume_text, top_n=12)
    jd_keywords = extract_top_keywords(jd_text, top_n=12)

    section_flags = detect_sections(resume_text)
    matched_skills = sorted(set(resume_skills).intersection(jd_skills))
    missing_skills = sorted(set(jd_skills).difference(resume_skills))

    scores = score_resume(
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        jd_keywords=jd_keywords,
        resume_text=resume_text,
        section_flags=section_flags,
    )

    suggestions = build_suggestions(
        resume_text=resume_text,
        jd_text=jd_text,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        jd_keywords=jd_keywords,
        section_flags=section_flags,
    )

    tailored_summary = build_tailored_summary(matched_skills, jd_keywords, section_flags)

    tool_trace = [
        {
            "tool": "Text extraction / cleaning",
            "input": "Resume text + Job description text",
            "output": "Normalized text ready for analysis",
        },
        {
            "tool": "Skill matcher",
            "input": "Predefined resume/job skill catalog",
            "output": f"{len(matched_skills)} matched skills and {len(missing_skills)} missing skills",
        },
        {
            "tool": "Keyword scorer",
            "input": "Top JD keywords vs resume keywords",
            "output": f"Keyword coverage: {scores['keyword_coverage']}%",
        },
        {
            "tool": "Resume scorer",
            "input": "Skills, sections, and action verbs",
            "output": f"Overall score: {scores['overall']}/100",
        },
        {
            "tool": "Suggestion generator",
            "input": "Detected gaps and missing sections",
            "output": "Actionable resume improvements and tailored summary",
        },
    ]

    return {
        "goal": goal,
        "plan": plan,
        "tool_trace": tool_trace,
        "scores": scores,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "jd_keywords": jd_keywords,
        "resume_keywords": resume_keywords,
        "section_flags": section_flags,
        "tailored_summary": tailored_summary,
        "suggestions": suggestions,
    }


def to_markdown_report(result: dict):
    lines = []
    lines.append("# Job Application Assistant Agent Report")
    lines.append("")
    lines.append("## Goal")
    lines.append(result["goal"])
    lines.append("")
    lines.append("## Plan")
    for i, item in enumerate(result["plan"], 1):
        lines.append(f"{i}. {item}")
    lines.append("")
    lines.append("## Final Output")
    lines.append(f"- Overall Score: {result['scores']['overall']}/100")
    lines.append(f"- Skill Coverage: {result['scores']['skill_coverage']}%")
    lines.append(f"- Keyword Coverage: {result['scores']['keyword_coverage']}%")
    lines.append("")
    lines.append("### Matched Skills")
    lines.append(", ".join(result["matched_skills"]) if result["matched_skills"] else "None detected")
    lines.append("")
    lines.append("### Missing Skills")
    lines.append(", ".join(result["missing_skills"]) if result["missing_skills"] else "No major missing skills detected")
    lines.append("")
    lines.append("### Tailored Summary")
    lines.append(result["tailored_summary"])
    lines.append("")
    lines.append("### Suggestions")
    for s in result["suggestions"]:
        lines.append(f"- {s}")
    return "\n".join(lines)


# -----------------------------
# UI helpers
# -----------------------------
def demo_resume_text():
    return """Aarav Sharma
Email: aarav@email.com | Phone: 9876543210

Professional Summary
Motivated B.Tech student with hands-on experience in Python, SQL, and data analysis. Built web applications using Streamlit and Flask.

Skills
Python, SQL, Pandas, NumPy, Streamlit, Flask, Git, GitHub, Communication, Teamwork

Projects
- Built a Student Performance Dashboard using Python and Streamlit.
- Developed a Resume Screening Tool to match resumes with job requirements.

Education
B.Tech Computer Engineering
"""


def demo_jd_text():
    return """Job Description: Data Analyst Intern

We are looking for a candidate with strong Python, SQL, Excel, and data visualization skills.
Responsibilities include analyzing datasets, creating dashboards, preparing reports, and collaborating with teams.
Preferred experience with Power BI, Pandas, and communication skills.
"""


def main():
    st.title("🧭 Job Application Assistant Agent")
    st.caption("Version 1 and Version 2: resume + job description analysis with goal → plan → tool use → final output")

    with st.sidebar:
        st.header("How this works")
        st.write("The agent reads the resume and job description, plans the analysis, uses tools to compare them, and returns a final report.")
        st.write("This is a workflow-based agent, not a trained model.")
        if st.button("Load demo data"):
            st.session_state["resume_text"] = demo_resume_text()
            st.session_state["jd_text"] = demo_jd_text()
            st.session_state["resume_file_loaded"] = ""
            st.session_state["jd_file_loaded"] = ""
            st.success("Demo data loaded.")

    tab1, tab2 = st.tabs(["Version 1 — Paste Text", "Version 2 — Upload Files"])

    with tab1:
        st.subheader("Paste Resume and Job Description")
        resume_text = st.text_area(
            "Resume text",
            value=st.session_state.get("resume_text", ""),
            height=260,
            placeholder="Paste the resume text here...",
            key="resume_text",
        )
        jd_text = st.text_area(
            "Job description text",
            value=st.session_state.get("jd_text", ""),
            height=260,
            placeholder="Paste the job description here...",
            key="jd_text",
        )

        if st.button("Analyze text inputs", type="primary"):
            result = run_agent(resume_text, jd_text)
            show_result(result)

    with tab2:
        st.subheader("Upload Resume and Job Description Files")
        resume_file = st.file_uploader(
            "Upload resume (.pdf or .txt)",
            type=["pdf", "txt"],
            key="resume_upload",
        )
        jd_file = st.file_uploader(
            "Upload job description (.pdf or .txt)",
            type=["pdf", "txt"],
            key="jd_upload",
        )

        if resume_file:
            st.session_state["resume_file_loaded"] = extract_text_from_upload(resume_file)
        if jd_file:
            st.session_state["jd_file_loaded"] = extract_text_from_upload(jd_file)

        resume_text_upload = st.text_area(
            "Extracted resume text",
            value=st.session_state.get("resume_file_loaded", ""),
            height=220,
        )
        jd_text_upload = st.text_area(
            "Extracted job description text",
            value=st.session_state.get("jd_file_loaded", ""),
            height=220,
        )

        if st.button("Analyze uploaded files", type="primary"):
            result = run_agent(resume_text_upload, jd_text_upload)
            show_result(result)


def show_result(result: dict):
    if "error" in result:
        st.error(result["error"])
        return

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Score", f"{result['scores']['overall']}/100")
    c2.metric("Skill Coverage", f"{result['scores']['skill_coverage']}%")
    c3.metric("Keyword Coverage", f"{result['scores']['keyword_coverage']}%")
    c4.metric("Matched Skills", len(result["matched_skills"]))

    st.subheader("Goal")
    st.write(result["goal"])

    st.subheader("Plan")
    for step in result["plan"]:
        st.write(f"• {step}")

    st.subheader("Tool Use")
    with st.expander("See how the agent used tools"):
        for item in result["tool_trace"]:
            st.markdown(
                f"**{item['tool']}**\n\n"
                f"- Input: {item['input']}\n"
                f"- Output: {item['output']}"
            )

    st.subheader("Final Output")
    st.markdown("### Matched Skills")
    st.write(", ".join(result["matched_skills"]) if result["matched_skills"] else "No major matches found.")

    st.markdown("### Missing Skills")
    st.write(", ".join(result["missing_skills"]) if result["missing_skills"] else "No major gaps detected.")

    st.markdown("### Tailored Summary")
    st.info(result["tailored_summary"])

    st.markdown("### Suggestions")
    for item in result["suggestions"]:
        st.write(f"• {item}")

    # -----------------------------
    # LLM Suggestions (Gemini)
    # -----------------------------
    st.markdown("### 🤖 AI-Powered Suggestions (LLM)")
    
    # Try to extract a simple job role from jd_text, or default to a generic one
    job_role = result.get("jd_keywords", ["Target Role"])[0].capitalize() if result.get("jd_keywords") else "Target Job Role"

    if result["missing_skills"]:
        with st.spinner("Generating AI suggestions using Gemini..."):
            llm_output = generate_llm_suggestions(
                result["missing_skills"],
                job_role
            )
        
        st.success("AI Suggestions Generated")
        st.write(llm_output)
    else:
        st.info("No major missing skills detected, so LLM suggestions not required.")

    with st.expander("Section audit"):
        st.json(result["section_flags"])

    with st.expander("Download report"):
        report = to_markdown_report(result)
        st.download_button(
            "Download markdown report",
            data=report,
            file_name="job_application_assistant_report.md",
            mime="text/markdown",
        )

    with st.expander("Raw result JSON"):
        st.code(json.dumps(result, indent=2), language="json")


if __name__ == "__main__":
    main()
