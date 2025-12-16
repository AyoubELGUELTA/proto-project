from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from typing import List, Dict, Any, cast
from normalize_llm_response import normalize_llm_content #to stringify the llm response



load_dotenv()



def create_ai_enhanced_summary(text: str, tables: list[str], images: list[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """
        
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

            # Build message content starting with text
        message_content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt_text}
        ]   
            
            # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(
            content=cast(List[Dict[str, Any]], message_content)  # type: ignore[arg-type], because everythings work fine
        )        
        response = llm.invoke([message])
        
        print(response.content)
        return normalize_llm_content(response.content)

        
    except Exception as e:
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary