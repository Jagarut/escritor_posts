import os
import ollama
from prompt_lib import PROMPT_LIBRARY
from mistralai import Mistral, UserMessage
from dotenv import load_dotenv
import requests
import json

load_dotenv()

# Initialize clients
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
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
    
    elif model.startswith("ollama"):
        
        
        print(f"Response generated by: {model}")
        print("================")
        # print(f"Prompt: {system_prompt}")
        # print("================")
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


# prompt = "Write a first-kiss scene between two rivals in a candlelit library."

# r = generate_text(prompt,  model="ollama/hermes3:3b")
# # r = generate_text(prompt)
# print(r)
