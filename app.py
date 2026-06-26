from __future__ import annotations

import json
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents import MultiAgentSoftwareEngineer
from src.evaluation import evaluate_pipeline
from src.report_content import PROJECT_REPORT
from src.storage import load_recent_runs, save_run


SAMPLE_REQUIREMENT = "Create a student attendance management system with login, attendance marking, reports, and admin dashboard."


st.set_page_config(
    page_title="Multi-Agent LLM for Autonomous Software Engineering",
    page_icon="ML",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1220px;
        }
        .hero {
            border: 1px solid #dbe4f0;
            background: linear-gradient(135deg, #ffffff 0%, #eef6ff 52%, #f4fff7 100%);
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 18px;
        }
        .hero h1 {
            margin: 0 0 10px 0;
            font-size: 2.25rem;
            color: #0f172a;
            letter-spacing: 0;
        }
        .hero p {
            margin: 0;
            color: #334155;
            font-size: 1.03rem;
            line-height: 1.55;
        }
        .metric-card {
            border: 1px solid #dbe4f0;
            background: #ffffff;
            border-radius: 8px;
            padding: 16px;
            min-height: 112px;
        }
        .metric-card .label {
            color: #64748b;
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .03em;
        }
        .metric-card .value {
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 750;
            margin-top: 8px;
        }
        .agent-row {
            border: 1px solid #dbe4f0;
            border-left: 5px solid #2563eb;
            background: #ffffff;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }
        .agent-row strong {
            color: #0f172a;
        }
        .agent-row span {
            color: #475569;
            font-size: .92rem;
        }
        .status {
            display: inline-block;
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: .78rem;
            font-weight: 700;
            margin-left: 8px;
        }
        .section-band {
            border: 1px solid #dbe4f0;
            background: #ffffff;
            border-radius: 8px;
            padding: 18px;
            margin: 12px 0;
        }
        .small-muted {
            color: #64748b;
            font-size: .92rem;
        }
        div[data-testid="stProgress"] > div > div > div > div {
            background-color: #2563eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_engine(provider: str) -> MultiAgentSoftwareEngineer:
    return MultiAgentSoftwareEngineer(provider)


def initialize_state() -> None:
    if "requirement" not in st.session_state:
        st.session_state.requirement = SAMPLE_REQUIREMENT
    if "result" not in st.session_state:
        st.session_state.result = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = None
    if "provider" not in st.session_state:
        st.session_state.provider = "Local Mock LLM"


def run_agents(requirement: str, show_progress: bool = True) -> None:
    engine = get_engine(st.session_state.provider)
    if show_progress:
        progress = st.progress(0)
        status = st.empty()
        steps = [
            "Requirement Analysis Agent is extracting features",
            "Task Planning Agent is decomposing work",
            "Code Generation Agent is preparing sample code",
            "Testing Agent is creating test cases",
            "Debugging Agent is scanning for issues",
            "Code Review Agent is checking quality",
            "Documentation Agent is producing final notes",
        ]
        for index, step in enumerate(steps, start=1):
            status.info(step)
            progress.progress(index / len(steps))
            time.sleep(0.12)
        status.success("All agents completed successfully.")
    result = engine.run(requirement)
    metrics = evaluate_pipeline(result)
    save_run(requirement, result, metrics)
    st.session_state.requirement = requirement
    st.session_state.result = result
    st.session_state.metrics = metrics


def ensure_result() -> None:
    if st.session_state.result is None:
        run_agents(st.session_state.requirement, show_progress=False)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Multi-Agent LLM for Autonomous Software Engineering</h1>
            <p>
                A Machine Learning and NLP based system where specialized LLM-style agents collaborate
                to transform natural-language requirements into analysis, plans, code, tests, reviews,
                documentation, and evaluation scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: dict[str, float]) -> None:
    cols = st.columns(3)
    for index, (label, value) in enumerate(metrics.items()):
        with cols[index % 3]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(value / 100)


def home_page() -> None:
    hero()
    cols = st.columns([1.15, 0.85])
    with cols[0]:
        st.subheader("Problem Statement")
        st.write(
            "Software projects usually need repeated manual effort for requirement understanding, planning, "
            "sample implementation, test design, review, and documentation. This project demonstrates how "
            "ML, NLP, and multi-agent LLM reasoning can automate those stages in a transparent workflow."
        )
        if st.button("Start New Requirement", type="primary", use_container_width=True):
            st.session_state.page = "Requirement Input"
            st.rerun()
    with cols[1]:
        st.subheader("Included ML Concepts")
        concepts = [
            "Natural Language Processing",
            "Large Language Model style reasoning",
            "Prompt engineering workflow",
            "Text classification",
            "Text summarization",
            "Information extraction",
            "Multi-agent task decomposition",
            "Evaluation metrics",
        ]
        for concept in concepts:
            st.write(f"- {concept}")

    st.subheader("Recent Runs")
    recent = load_recent_runs()
    if recent:
        st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
    else:
        st.info("No saved runs yet. Submit a requirement to create the first result.")


def requirement_input_page() -> None:
    st.title("Requirement Input")
    st.write("Enter a software requirement in natural language. The local ML pipeline will classify, extract, and route it through seven agents.")
    provider = st.selectbox(
        "LLM Provider",
        ["Local Mock LLM", "OpenAI", "Hugging Face"],
        index=["Local Mock LLM", "OpenAI", "Hugging Face"].index(st.session_state.provider),
        help="OpenAI uses OPENAI_API_KEY. Hugging Face uses HF_API_TOKEN. Local Mock LLM always works offline.",
    )
    st.session_state.provider = provider
    with st.expander("Provider Setup Status"):
        st.json(get_engine(provider).llm.status())

    requirement = st.text_area(
        "Software Requirement",
        value=st.session_state.requirement,
        height=170,
        placeholder="Example: Build an online library management system with book issue, return, login, reports, and admin dashboard.",
    )
    cols = st.columns([0.7, 0.3])
    with cols[0]:
        submitted = st.button("Run Multi-Agent Workflow", type="primary", use_container_width=True)
    with cols[1]:
        sample = st.button("Use Sample Requirement", use_container_width=True)

    if sample:
        st.session_state.requirement = SAMPLE_REQUIREMENT
        st.rerun()
    if submitted:
        if len(requirement.strip()) < 20:
            st.error("Please enter a more detailed requirement with at least 20 characters.")
        else:
            run_agents(requirement.strip())
            st.success("Workflow completed. Open the Output and Evaluation pages for the generated results.")


def workflow_page() -> None:
    ensure_result()
    st.title("Agent Workflow")
    st.write("Each agent performs one specialized software engineering task and passes structured output to the next agent.")
    engine = get_engine(st.session_state.provider)
    for step, item in enumerate(engine.workflow_results(st.session_state.result), start=1):
        st.markdown(
            f"""
            <div class="agent-row">
                <strong>Step {step}: {item.name}</strong><span class="status">{item.status}</span><br>
                <span>{item.role}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"View {item.name} Output"):
            st.json(item.output)


def output_page() -> None:
    ensure_result()
    result = st.session_state.result
    analysis = result["analysis"]
    st.title("Final Output")
    st.subheader("Requirement Summary")
    st.info(analysis["summary"])
    if analysis.get("llm_enhanced_summary"):
        st.success(f"LLM Enhanced Summary: {analysis['llm_enhanced_summary']}")
    with st.expander("Provider Status"):
        st.json(result.get("llm_provider", {}))

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "Download Output JSON",
            data=json.dumps({"result": result, "metrics": st.session_state.metrics}, indent=2),
            file_name="multi_agent_output.json",
            mime="application/json",
            use_container_width=True,
        )
    with export_col2:
        st.download_button(
            "Download Output Markdown",
            data=result_to_markdown(result, st.session_state.metrics),
            file_name="multi_agent_output.md",
            mime="text/markdown",
            use_container_width=True,
        )

    tab_names = [
        "Analysis",
        "Task Breakdown",
        "Generated Code",
        "Test Cases",
        "Execution",
        "Debugging",
        "Code Review",
        "Documentation",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Extracted Features")
            st.table(pd.DataFrame({"features": analysis["features"]}))
            st.write("Modules")
            st.table(pd.DataFrame({"modules": analysis["modules"]}))
        with col2:
            st.write("ML/NLP Details")
            st.json(
                {
                    "domain": analysis["domain"],
                    "keywords": analysis["keywords"],
                    "dataset_match": analysis["dataset_match"],
                    "entities": analysis["entities"],
                }
            )
    with tabs[1]:
        st.dataframe(pd.DataFrame(result["plan"]["tasks"]), use_container_width=True, hide_index=True)
        st.write("Architecture")
        st.write(" -> ".join(result["plan"]["architecture"]))
    with tabs[2]:
        st.code(result["code"], language="python")
    with tabs[3]:
        st.dataframe(pd.DataFrame(result["tests"]), use_container_width=True, hide_index=True)
    with tabs[4]:
        st.metric("Generated Code Execution", result["execution"]["overall_status"])
        st.dataframe(pd.DataFrame(result["execution"]["checks"]), use_container_width=True, hide_index=True)
    with tabs[5]:
        st.json(result["debugging"])
    with tabs[6]:
        st.json(result["review"])
    with tabs[7]:
        for title, content in result["documentation"].items():
            st.markdown(f"**{title.replace('_', ' ').title()}**")
            st.write(content)


def evaluation_page() -> None:
    ensure_result()
    metrics = st.session_state.metrics
    st.title("Evaluation")
    st.write("The evaluation module calculates transparent ML demo metrics for response quality and task completion.")
    render_metric_cards(metrics)

    metric_df = pd.DataFrame({"Metric": list(metrics.keys()), "Score": list(metrics.values())})
    fig = px.bar(metric_df, x="Metric", y="Score", range_y=[0, 100], text="Score", color="Score", color_continuous_scale="Blues")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=35, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metric Explanation")
    st.dataframe(
        pd.DataFrame(
            [
                {"Metric": "Accuracy Score", "Meaning": "Average of syntax, completeness, relevance, quality, readability, and completion scores."},
                {"Metric": "Completeness Score", "Meaning": "Measures whether enough features, modules, and development tasks were generated."},
                {"Metric": "Relevance Score", "Meaning": "Uses TF-IDF cosine similarity against the demo requirement dataset."},
                {"Metric": "Code Quality Score", "Meaning": "Checks syntax and structural code elements such as classes, functions, types, and validation."},
                {"Metric": "Readability Score", "Meaning": "Estimates clarity from concise length and structured generated code."},
                {"Metric": "Task Completion Score", "Meaning": "Measures generated test coverage and documentation section completion."},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def about_page() -> None:
    st.title("About Project")
    st.write("Complete project report content for presentation and final-year submission.")
    for title, content in PROJECT_REPORT.items():
        st.markdown(f'<div class="section-band"><h3>{title}</h3><p>{content}</p></div>', unsafe_allow_html=True)

    st.subheader("Sample Dataset")
    dataset = pd.read_csv("data/sample_requirements.csv")
    st.caption(f"Dataset contains {len(dataset)} demonstration requirements across {dataset['domain'].nunique()} domains.")
    st.dataframe(dataset, use_container_width=True, hide_index=True)


def sidebar() -> str:
    st.sidebar.title("Project Navigation")
    pages = ["Home", "Requirement Input", "Agent Workflow", "Output", "Evaluation", "About Project"]
    current = st.session_state.get("page", "Home")
    page = st.sidebar.radio("Pages", pages, index=pages.index(current))
    st.session_state.page = page
    st.sidebar.divider()
    st.sidebar.caption("Offline compatible: no paid API key required.")
    st.sidebar.caption("ML: TF-IDF, Naive Bayes, cosine similarity, rule-based NLP.")
    st.sidebar.caption(f"LLM provider: {st.session_state.get('provider', 'Local Mock LLM')}")
    return page


def main() -> None:
    inject_css()
    initialize_state()
    page = sidebar()
    if page == "Home":
        home_page()
    elif page == "Requirement Input":
        requirement_input_page()
    elif page == "Agent Workflow":
        workflow_page()
    elif page == "Output":
        output_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "About Project":
        about_page()


if __name__ == "__main__":
    main()
def result_to_markdown(result: dict, metrics: dict[str, float]) -> str:
    analysis = result["analysis"]
    lines = [
        "# Multi-Agent LLM Generated Output",
        "",
        "## Requirement Summary",
        analysis["summary"],
        "",
        "## LLM Enhanced Summary",
        analysis.get("llm_enhanced_summary", ""),
        "",
        "## Extracted Features",
        *[f"- {feature}" for feature in analysis["features"]],
        "",
        "## Task Breakdown",
        *[f"- **{task['phase']}**: {task['task']}" for task in result["plan"]["tasks"]],
        "",
        "## Generated Code",
        "```python",
        result["code"],
        "```",
        "",
        "## Test Cases",
        *[f"- **{item['type']}**: {item['case']} Expected: {item['expected']}" for item in result["tests"]],
        "",
        "## Execution Checks",
        *[f"- {item['check']}: {item['status']} - {item['detail']}" for item in result["execution"]["checks"]],
        "",
        "## Evaluation Scores",
        *[f"- {key}: {value:.2f}%" for key, value in metrics.items()],
    ]
    return "\n".join(lines)

