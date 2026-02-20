import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

load_dotenv()


class SynthesizerAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.4,  # higher for more creative synthesis
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        prompt_file = Path(__file__).parent.parent / "prompts" / "synthesizer_prompt.txt"
        return prompt_file.read_text()
    
    def _format_summaries_for_prompt(self, summaries: List[Dict]) -> str:
        formatted = []
        
        for item in summaries:
            keyword = item['keyword']
            formatted.append(f"\n=== Research Area: {keyword} ===\n")
            
            for summary in item['summaries']:
                source_type = summary['source_type']
                title = summary['title']
                key_points = summary['key_points']
                
                formatted.append(f"\nSource ({source_type}): {title}")
                formatted.append("Key Points:")
                for point in key_points:
                    formatted.append(f"  - {point}")
                formatted.append("")
        
        return "\n".join(formatted)
    
    def synthesize(self, summaries: List[Dict], topic: str) -> Dict[str, str]:
        print("\n" + "="*80)
        print("SYNTHESIZER AGENT")
        print("="*80)
        print(f"\nSynthesizing research report for: {topic}")
        
        formatted_summaries = self._format_summaries_for_prompt(summaries)
        
        prompt = self.prompt_template.format(
            topic=topic,
            summaries=formatted_summaries
        )
        
        try:
            print("  Generating synthesis...")
            response = self.llm.invoke(prompt)
            report_text = response.content if hasattr(response, 'content') else str(response)
            
            print(f"✓ Synthesis complete ({len(report_text)} characters)")
            print("="*80)
            
            return {
                'topic': topic,
                'report_text': report_text,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"✗ Error during synthesis: {e}")
            return {
                'topic': topic,
                'report_text': f"Error generating synthesis: {str(e)}",
                'generated_at': datetime.now().isoformat()
            }
    
    def generate_pdf(self, synthesis: Dict[str, str], output_path: str):
        print("\n" + "="*80)
        print("GENERATING PDF REPORT")
        print("="*80)
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#1a1a1a',
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            fontName='Times-Roman' 
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#666666',
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        )

        story = []

        title = Paragraph(f"Research Report:<br/>{synthesis['topic']}", title_style)
        story.append(title)

        date_str = datetime.now().strftime("%B %d, %Y")
        subtitle = Paragraph(f"Generated on {date_str}", subtitle_style)
        story.append(subtitle)
        story.append(Spacer(1, 0.2*inch))

        from reportlab.platypus import HRFlowable
        story.append(HRFlowable(width="100%", thickness=1, color='#cccccc'))
        story.append(Spacer(1, 0.3*inch))
        
        report_text = synthesis['report_text']
        paragraphs = report_text.split('\n\n')
        
        for para_text in paragraphs:
            if para_text.strip():
                if para_text.strip().isupper() or para_text.strip().startswith('#'):
                    heading_text = para_text.strip().replace('#', '').strip()
                    heading = Paragraph(heading_text, styles['Heading2'])
                    story.append(Spacer(1, 0.2*inch))
                    story.append(heading)
                    story.append(Spacer(1, 0.1*inch))
                else:
                    para = Paragraph(para_text.strip(), body_style)
                    story.append(para)

        story.append(Spacer(1, 0.5*inch))
        
        try:
            doc.build(story)
            print(f"PDF generated: {output_path}")
            print("="*80)
        except Exception as e:
            print(f"Error generating PDF: {e}")
            print("="*80)
            raise