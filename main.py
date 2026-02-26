import streamlit as st
import json
from pathlib import Path
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from agents.synthesizer import SynthesizerAgent
from PIL import Image

icon = Image.open("icon3.png")

gradient_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #373737, 
        #2a2a2a, 
        #1e1e1e 
    );
}
</style>
"""

st.markdown(gradient_css, unsafe_allow_html=True)

st.set_page_config(
    page_title="Research Planner Agent",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;0,900;1,300;1,400;1,700;1,900&family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

LOGO_URL = "https://registry.npmmirror.com/@lobehub/icons-static-png/1.65.0/files/dark/grok.png"
st.logo(LOGO_URL, size="large")

st.title("**Multi-Agent Research Planner**", text_alignment="center")
st.markdown("Generate comprehensive research reports using AI agents", text_alignment="center")

if 'stage' not in st.session_state:
    st.session_state.stage = 'input'
    st.session_state.topic = ''
    st.session_state.keywords = []
    st.session_state.retry_count = 0
    st.session_state.retrieval_results = []
    st.session_state.summaries = []
    st.session_state.synthesis = {}

def reset_pipeline():
    st.session_state.stage = 'input'
    st.session_state.topic = ''
    st.session_state.keywords = []
    st.session_state.retry_count = 0
    st.session_state.retrieval_results = []
    st.session_state.summaries = []
    st.session_state.synthesis = {}

if st.session_state.stage == 'input':
    st.markdown("### Enter Research Topic", text_alignment="center")
    topic = st.text_input("Research Topic", placeholder="e.g., Vision Transformers", key="topic_input", label_visibility="collapsed")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Start", type="secondary", disabled=not topic):
            st.session_state.topic = topic
            st.session_state.stage = 'planning'
            st.rerun()

elif st.session_state.stage == 'planning':
    st.markdown("---")
    st.markdown(f"### Topic: {st.session_state.topic}")
    st.markdown("---")
    
    if not st.session_state.keywords:
        with st.spinner("Planner Agent: Generating keywords..."):
            planner = PlannerAgent()
            result = planner.generate_keywords(
                st.session_state.topic, 
                retry_count=st.session_state.retry_count
            )
            st.session_state.keywords = result['keywords']
    
    st.success("Keywords Generated")
    
    st.markdown("#### Generated Keywords:")
    cols = st.columns(2)
    for i, kw in enumerate(st.session_state.keywords):
        with cols[i % 2]:
            st.info(f"**{i+1}.** {kw}")
    
    if st.session_state.retry_count > 0:
        st.caption(f"(Retry attempt {st.session_state.retry_count}/1)")
    
    st.markdown("---")
    st.markdown("#### Options:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Accept Keywords", type="primary", use_container_width=True):
            st.session_state.stage = 'retrieving'
            st.rerun()
    with col2:
        if st.button("Retry (More Specific)", use_container_width=True, disabled=st.session_state.retry_count >= 1):
            st.session_state.retry_count += 1
            st.session_state.keywords = []
            st.rerun()
    with col3:
        if st.button("Manual Edit", use_container_width=True):
            st.session_state.stage = 'manual_edit'
            st.rerun()
    
    if st.session_state.retry_count >= 1:
        st.caption("No more retries available")

elif st.session_state.stage == 'manual_edit':
    st.markdown("---")
    st.markdown(f"### Topic: {st.session_state.topic}")
    st.markdown("---")
    st.markdown("#### Manual Edit - Replace a Keyword")
    
    st.markdown("**Current Keywords:**")
    cols = st.columns(2)
    for i, kw in enumerate(st.session_state.keywords):
        with cols[i % 2]:
            st.info(f"**{i+1}.** {kw}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        keyword_index = st.selectbox(
            "Which keyword to replace?",
            options=range(len(st.session_state.keywords)),
            format_func=lambda x: f"{x+1}. {st.session_state.keywords[x]}"
        )
    with col2:
        new_keyword = st.text_input("Enter new keyword:", key="new_keyword_input")
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Replace Keyword", type="primary", use_container_width=True, disabled=not new_keyword):
            planner = PlannerAgent()
            updated_keywords = planner.replace_keyword(
                st.session_state.keywords.copy(), 
                keyword_index, 
                new_keyword
            )
            st.session_state.keywords = updated_keywords
            st.success(f"Replaced keyword {keyword_index+1}")
            st.rerun()
    with col2:
        if st.button("Review Keywords", use_container_width=True):
            st.session_state.stage = 'planning'
            st.rerun()
    with col3:
        if st.button("Cancel", use_container_width=True):
            st.session_state.stage = 'planning'
            st.rerun()

elif st.session_state.stage == 'retrieving':
    st.markdown("---")
    st.markdown(f"### Topic: {st.session_state.topic}")
    st.markdown("---")
    
    st.markdown("#### Keywords:")
    cols = st.columns(2)
    for i, kw in enumerate(st.session_state.keywords):
        with cols[i % 2]:
            st.info(f"**{i+1}.** {kw}")
    
    st.markdown("---")
    
    if not st.session_state.retrieval_results:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        retriever = RetrieverAgent()
        total = len(st.session_state.keywords)
        
        results = []
        for i, keyword in enumerate(st.session_state.keywords, 1):
            status_text.markdown(f"**Retrieving sources for:** `{keyword}` **[{i}/{total}]**")
            progress_bar.progress(i / total)
            
            wiki_result = retriever._fetch_wikipedia(keyword)
            import time
            time.sleep(1)
            arxiv_results = retriever._fetch_arxiv(keyword, max_results=1)
            
            results.append({
                "keyword": keyword,
                "wikipedia": wiki_result,
                "arxiv_papers": arxiv_results
            })
            time.sleep(1)
        
        st.session_state.retrieval_results = results
        status_text.empty()
        progress_bar.empty()
    
    st.success("Retrieval Complete")
    
    st.markdown("---")
    st.markdown("#### Retrieved Sources:")
    
    wiki_tab, arxiv_tab = st.tabs(["Wikipedia", "arXiv Papers"])
    
    with wiki_tab:
        wiki_count = 0
        for result in st.session_state.retrieval_results:
            wiki = result['wikipedia']
            if wiki['title']:
                wiki_count += 1
                st.markdown(f"**{wiki_count}.** [{wiki['title']}]({wiki['url']})")
        if wiki_count == 0:
            st.info("No Wikipedia sources found")
    
    with arxiv_tab:
        arxiv_count = 0
        for result in st.session_state.retrieval_results:
            for paper in result['arxiv_papers']:
                arxiv_count += 1
                st.markdown(f"**{arxiv_count}.** [{paper['title']}]({paper['url']})")
                st.caption(f"Published: {paper['published']}")
        if arxiv_count == 0:
            st.info("No arXiv papers found")
    
    st.info(f"**Total:** {wiki_count} Wikipedia articles, {arxiv_count} arXiv papers")
    
    st.markdown("---")
    if st.button("Continue to Summarization", type="primary"):
        st.session_state.stage = 'summarizing'
        st.rerun()

elif st.session_state.stage == 'summarizing':
    st.markdown("---")
    st.markdown(f"### Topic: {st.session_state.topic}")
    st.markdown("---")
    
    if not st.session_state.summaries:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        summarizer = SummarizerAgent()
        
        total_sources = sum(
            1 + len(doc.get('arxiv_papers', []))
            for doc in st.session_state.retrieval_results
        )
        
        current = 0
        all_summaries = []
        
        for doc in st.session_state.retrieval_results:
            keyword = doc['keyword']
            summaries_for_keyword = []
            
            status_text.markdown(f"**Processing keyword:** `{keyword}`")
            
            wiki = doc.get('wikipedia', {})
            if wiki.get('title'):
                current += 1
                status_text.markdown(f"**[{current}/{total_sources}]** Summarizing Wikipedia: *{wiki['title'][:50]}...*")
                progress_bar.progress(current / total_sources)
                
                wiki_summary = summarizer._summarize_source(
                    source_type="wikipedia",
                    title=wiki['title'],
                    content=wiki.get('content', ''),
                    url=wiki.get('url', '')
                )
                summaries_for_keyword.append(wiki_summary)
            
            arxiv_papers = doc.get('arxiv_papers', [])
            for paper in arxiv_papers:
                current += 1
                status_text.markdown(f"**[{current}/{total_sources}]** Summarizing arXiv: *{paper['title'][:50]}...*")
                progress_bar.progress(current / total_sources)
                
                arxiv_summary = summarizer._summarize_source(
                    source_type="arxiv",
                    title=paper['title'],
                    content=paper.get('abstract', ''),
                    url=paper.get('url', '')
                )
                summaries_for_keyword.append(arxiv_summary)
            
            all_summaries.append({
                "keyword": keyword,
                "summaries": summaries_for_keyword
            })
        
        st.session_state.summaries = all_summaries
        status_text.empty()
        progress_bar.empty()
    
    total_summarized = sum(len(item['summaries']) for item in st.session_state.summaries)
    st.success(f"Summarization Complete - {total_summarized} sources summarized")
    
    st.markdown("---")
    st.markdown("#### Summary Preview:")
    for item in st.session_state.summaries:
        with st.expander(f"**{item['keyword']}** ({len(item['summaries'])} summaries)", expanded=False):
            for summary in item['summaries']:
                st.markdown(f"**{summary['source_type'].upper()}: {summary['title']}**")
                for point in summary['key_points']:
                    st.markdown(f"- {point}")
                st.markdown("---")
    
    st.markdown("---")
    if st.button("Continue to Synthesis", type="primary"):
        st.session_state.stage = 'synthesizing'
        st.rerun()

elif st.session_state.stage == 'synthesizing':
    st.markdown("---")
    st.markdown(f"### Topic: {st.session_state.topic}")
    st.markdown("---")
    
    if not st.session_state.synthesis:
        with st.spinner("Synthesizer Agent: Combining all summaries into final report..."):
            synthesizer = SynthesizerAgent()
            synthesis = synthesizer.synthesize(
                st.session_state.summaries,
                st.session_state.topic
            )
            st.session_state.synthesis = synthesis
    
    st.success(f"Synthesis Complete ({len(st.session_state.synthesis['report_text'])} characters)")
    
    st.markdown("---")
    st.markdown("#### Report Preview:")
    preview_text = st.session_state.synthesis['report_text'][:500] + "..."
    st.text_area("First 500 characters:", preview_text, height=200, disabled=True)
    
    st.markdown("---")
    st.markdown("#### Download Options:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF..."):
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                
                safe_filename = "".join(c for c in st.session_state.topic if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_filename = safe_filename.replace(' ', '_')[:50]
                pdf_path = output_dir / f"{safe_filename}_report.pdf"
                
                synthesizer = SynthesizerAgent()
                synthesizer.generate_pdf(st.session_state.synthesis, str(pdf_path))
                
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{safe_filename}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.success("PDF Generated Successfully!")
    
    with col2:
        synthesis_json = json.dumps(st.session_state.synthesis, indent=2)
        st.download_button(
            label="Download Synthesis JSON",
            data=synthesis_json,
            file_name=f"{st.session_state.topic.replace(' ', '_')}_synthesis.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    if st.button("Start New Research", use_container_width=True):
        reset_pipeline()
        st.rerun()

st.sidebar.markdown("### Pipeline Status")
st.sidebar.markdown("")

stages = [
    ('input', 'Input Topic'),
    ('planning', 'Planning Keywords'),
    ('manual_edit', 'Editing Keywords'),
    ('retrieving', 'Retrieving Sources'),
    ('summarizing', 'Summarizing Sources'),
    ('synthesizing', 'Synthesizing Report')
]

for stage_key, stage_label in stages:
    if st.session_state.stage == stage_key:
        st.sidebar.markdown(f"**{stage_label}** (Active)")
    else:
        st.sidebar.markdown(f"   {stage_label}")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("Multi-agent system for automated research report generation")
st.sidebar.markdown("")
st.sidebar.markdown("**Agents:**")
st.sidebar.markdown("- **Planner** - Llama 3.3")
st.sidebar.markdown("- **Retriever** - Wiki + arXiv")
st.sidebar.markdown("- **Summarizer** - Gemini 2.5 Flash")
st.sidebar.markdown("- **Synthesizer** - GPT-Oss(120b)")