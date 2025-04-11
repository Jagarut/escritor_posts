import os
import re 
import ollama
import lmstudio as lms
from fpdf import FPDF
# from prompt_lib import PROMPT_LIBRARY
from mistralai import Mistral
from groq import Groq
from dotenv import load_dotenv
import requests
import json

load_dotenv()

# Initialize clients
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def generate_text(
    prompt: str,
    model: str = "mistral-small-latest",  # Default to Mistral
    temperature: float = 0.7,
    system_prompt: str = None
) -> str:
    """
    Generates text using Mistral API or local Ollama models.
    
    Args:
        prompt (str): Input prompt
        model (str): One of:
            - "mistral-small-latest" | "mistral-medium-latest" (Mistral API)
            - "ollama:mistral" | "ollama:llama2" (Ollama models)
        temperature (float): Creativity control (0-1)
        system_prompt (str): System message to guide the model's behavior
    Returns:
        str: Generated text
    """
    if model.startswith("mistral"):
        print(f"Response generated by: {model}")
        # Mistral API implementation
        response = mistral_client.chat.complete(
            model=model,
            # messages=[ChatMessage(role="user", content=prompt)],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    elif model.startswith("groq"):
        print(f"Response generated by: {model}")
        print("================")
        m = model.split("/")
        model = m[1]
        if model.startswith("meta-llama"):
            model = "/".join(m[1:])
        # Groq API implementation
        try:
            chat_completion = groq_client.chat.completions.create(
                model=model,
                # messages=messages,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user", 
                        "content": prompt,
                    }
                ],      
                temperature=temperature,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error in chat_completions_create: {e}")
    
    elif model.startswith("ollama"):
        print(f"Response generated by: {model}")
        print("================")

        model = model.split("/")[1]
        
        # Ollama local model implementation
        try:
            response = ollama.chat(
                model=model, 
                messages=[
                    {
                        'role': 'system', 
                        'content': system_prompt, 
                    }, 
                    {
                        'role': 'user', 
                        'content': f"{prompt}", 
                    }, 
                ]
            )
            # print(type(response['message']['content']))
            # print("================")
            return(response['message']['content'])
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Ollama connection failed: {str(e)}")
        
    elif model.startswith("LMstudio"):
        
        
        print(f"Response generated by: {model}")
        print("================")

        model = model.split("/")[1]
        # LMstudio local model implementation
        try:
            model = lms.llm(model)

            response = model.respond({
                "messages":[
                    {
                        'role': 'system', 
                        'content': system_prompt, 
                    }, 
                    {
                        'role': 'user', 
                        'content': f"{prompt}", 
                    }, 
                ]
            },
            config={"temperature": temperature}                         
            )
            return str(response)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"LMstudio connection failed: {str(e)}")
    
    else:
        raise ValueError(f"Unsupported model: {model}")


def refine_text(
    original_text: str,
    user_feedback: str,
    model: str = "mistral-small-latest",
    **kwargs
) -> str:
    """
    Refines text using Mistral or Ollama.
    
    Args:
        original_text (str): Text to refine
        user_feedback (str): Instructions for refinement
        model (str): Which LLM to use
        **kwargs: Additional model-specific args
    
    Returns:
        str: Refined text
    """
    prompt = f"""
    Refine this text based on the feedback below.
    Return ONLY the refined text, no additional commentary.
    
    Feedback: {user_feedback}
    ---
    Text to refine:
    {original_text}
    """
    return generate_text(prompt, model=model, **kwargs)


def split_into_paragraphs(text):
    """Split text into paragraphs while preserving empty lines as paragraph separators"""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    # The if p.strip() condition is necessary if you want to ensure that the resulting list does not contain any empty paragraphs. 
    # If you are okay with having empty paragraphs in the list, you can omit the condition
    return [p.strip() for p in paragraphs if p.strip()]

def join_paragraphs(paragraphs):
    """Join paragraphs with double newlines"""
    return '\n\n'.join(paragraphs)

def regenerate_paragraph(paragraph, instruction=None, context=None, model=None, system_prompt=None, temperature=0.7):
    """
    Regenerate a single paragraph with optional instructions and context
    
    Args:
        paragraph (str): The paragraph to regenerate
        instruction (str): How to modify the paragraph
        context (dict): Surrounding paragraphs for context
        model (str): Model identifier
        system_prompt (str): System message to guide the model
        temperature (float): Creativity parameter
    
    Returns:
        str: Regenerated paragraph
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    prompt = f"""Please regenerate the following paragraph{' ' + instruction if instruction else ''}:
    
Paragraph to regenerate:
{paragraph}
"""
    
    if context:
        if context.get("previous_paragraphs"):
            prompt += f"\nPrevious context:\n{'\n'.join(context['previous_paragraphs'])}\n"
        if context.get("next_paragraphs"):
            prompt += f"\nFollowing context:\n{'\n'.join(context['next_paragraphs'])}\n"
    
    prompt += "\nPlease provide only the regenerated paragraph, without any additional commentary or markup."
    
    messages.append({"role": "user", "content": prompt})
    
    # Call your AI model here - this will depend on your specific implementation
    response = generate_text(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt
    )
    
    # Clean up the response to ensure we only get the paragraph
    return response.strip()



# def generate_pdf(story_content, title="AI Generated Story", author="AI Story Writer"):
#     """Generate a PDF document from the story content"""
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
    
#     # Add a page
#     pdf.add_page()
    
#     # Set font for title
#     pdf.set_font("Arial", 'B', 16)
#     pdf.cell(0, 10, title, 0, 1, 'C')
    
#     # Set font for author
#     pdf.set_font("Arial", 'I', 12)
#     pdf.cell(0, 10, f"by {author}", 0, 1, 'C')
    
#     # Add space
#     pdf.ln(10)
    
#     # Set font for content
#     pdf.set_font("Arial", '', 12)
    
#     # Add story content with proper paragraph handling
#     for paragraph in story_content.split('\n\n'):
#         pdf.multi_cell(0, 8, paragraph.strip())
#         pdf.ln(5)  # Add space between paragraphs
    
#     return pdf

from fpdf import FPDF
from io import BytesIO

def generate_pdf(story_content, title="AI Generated Story", author="AI Story Writer"):
    """Generate PDF with Unicode support"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add DejaVu font (or any other Unicode font)
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        pdf.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
        font_family = 'DejaVu'
    except:
        # Fallback to Arial Unicode if DejaVu not available
        try:
            pdf.add_font('ArialUnicode', '', 'arial-unicode-ms.ttf', uni=True)
            font_family = 'ArialUnicode'
        except:
            font_family = 'Arial'
            print("Special characters may not display correctly - install DejaVu or Arial Unicode fonts for full support")
    
    # Add a page
    pdf.add_page()
    
    # Title
    pdf.set_font(font_family, 'B', 16)
    pdf.cell(0, 10, title, 0, 1, 'C')
    
    # Author
    pdf.set_font(font_family, 'I', 12)
    pdf.cell(0, 10, f"by {author}", 0, 1, 'C')
    pdf.ln(10)
    
    # Content
    pdf.set_font(font_family, '', 12)
    for paragraph in story_content.split('\n\n'):
        pdf.multi_cell(0, 8, paragraph.strip())
        pdf.ln(5)
    
    return pdf
                        
# Create PDF generation function if not already imported
# def generate_pdf(text, title, author):
    
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
    
#     # Set font for title
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, title, ln=True, align="C")
    
#     # Add author
#     pdf.set_font("Arial", "I", 12)
#     pdf.cell(0, 10, f"By: {author}", ln=True, align="C")
#     pdf.ln(5)
    
#     # Set font for body text
#     pdf.set_font("Arial", "", 12)
    
#     # Split text into paragraphs and add to PDF
#     paragraphs = text.split("\n\n")
#     for paragraph in paragraphs:
#         pdf.multi_cell(0, 10, paragraph)
#         pdf.ln(5)
        
#     return pdf
# prompt = "Write a first-kiss scene between two rivals in a candlelit library."

# r = generate_text(prompt,  model="ollama/hermes3:3b")
# # r = generate_text(prompt)
# print(r)
