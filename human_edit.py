import streamlit as st
from prompt_lib import PROMPT_LIBRARY
from story_manager import StoryManager
from utils import generate_text, refine_text, split_into_paragraphs, join_paragraphs
import re

# Available models with proper naming
MODELS = {
    "Mistral Small": "mistral-small-latest",
    "Mistral Medium": "mistral-medium-latest",
    "Ollama Mistral": "ollama/mistral",
    "Ollama Hermes 3": "ollama/hermes3:3b",
    "Ollama Llama3.2": "ollama/llama3.2:3b"
}

# System prompt options
SYSTEM_PROMPTS = PROMPT_LIBRARY

def main():
    st.set_page_config(page_title="AI Story Writer", layout="wide")
    st.title("📖 AI Story Writer")
    st.write("Generate and refine stories using Mistral or local Ollama models")

    # Initialize session state
    if "story_manager" not in st.session_state:
        st.session_state.story_manager = StoryManager()
        st.session_state.selected_model = "mistral-small-latest"
        st.session_state.temperature = 0.7
        st.session_state.system_prompt = SYSTEM_PROMPTS["Obedient AI"]
        st.session_state.editing_paragraph = None
        st.session_state.edited_paragraphs = []

    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        current_model = st.session_state.selected_model
        display_name = next(k for k, v in MODELS.items() if v == current_model)
        selected_display = st.selectbox(
            "AI Model",
            options=list(MODELS.keys()),
            index=list(MODELS.values()).index(current_model),
            key="model_select"
        )
        st.session_state.selected_model = MODELS[selected_display]
        
        # System prompt selection
        selected_prompt = st.selectbox(
            "Writing Style",
            options=list(SYSTEM_PROMPTS.keys()),
            index=list(SYSTEM_PROMPTS.values()).index(st.session_state.system_prompt),
            key="prompt_select"
        )
        st.session_state.system_prompt = SYSTEM_PROMPTS[selected_prompt]
        
        # Temperature control
        st.session_state.temperature = st.slider(
            "Creativity (temperature)",
            0.0, 1.0, st.session_state.temperature, 0.1,
            key="temp_slider"
        )
        
        # Model info
        if st.session_state.selected_model.startswith("ollama"):
            st.info("Using local Ollama model. Ensure the model is pulled and Ollama is running.")
        
        # Version history
        st.header("Version History")
        if st.session_state.story_manager.versions:
            for i, version in enumerate(st.session_state.story_manager.get_version_history()):
                with st.expander(f"v{i+1}: {version['feedback'][:50]}..."):
                    st.caption(version["feedback"])
                    st.code(version["text"][:200] + ("..." if len(version["text"]) > 200 else ""))

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        # First draft generation
        if not st.session_state.story_manager.versions:
            st.subheader("Create New Story")
            user_prompt = st.text_area(
                "Story prompt",
                "A detective in 2050 solves a murder using AI.",
                height=150
            )
            
            if st.button("Generate First Draft", use_container_width=True):
                with st.spinner(f"Generating with {selected_display}..."):
                    try:
                        draft = generate_text(
                            prompt=f"Write a short story about: {user_prompt}",
                            model=st.session_state.selected_model,
                            temperature=st.session_state.temperature,
                            system_prompt=st.session_state.system_prompt
                        )
                        st.session_state.story_manager.add_version(draft, "Initial draft")
                        st.session_state.edited_paragraphs = split_into_paragraphs(draft)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")

        # Editing interface
        else:
            st.subheader("Current Draft")
            latest_version = st.session_state.story_manager.get_latest_version()
            
            # Split into paragraphs for editing
            if not st.session_state.edited_paragraphs:
                st.session_state.edited_paragraphs = split_into_paragraphs(latest_version)
            
            # Display each paragraph with edit button
            for i, paragraph in enumerate(st.session_state.edited_paragraphs):
                with st.container():
                    cols = st.columns([4, 1])
                    with cols[0]:
                        if st.session_state.editing_paragraph == i:
                            edited = st.text_area(
                                f"Paragraph {i+1}",
                                paragraph,
                                key=f"edit_{i}",
                                height=150
                            )
                            if st.button("Save", key=f"save_{i}"):
                                st.session_state.edited_paragraphs[i] = edited
                                st.session_state.editing_paragraph = None
                                st.rerun()
                        else:
                            st.write(paragraph)
                    with cols[1]:
                        if st.session_state.editing_paragraph != i:
                            if st.button("✏️ Edit", key=f"edit_btn_{i}"):
                                st.session_state.editing_paragraph = i
                                st.rerun()
            
            # Save edited version
            if st.button("💾 Save Edited Version", use_container_width=True):
                edited_text = join_paragraphs(st.session_state.edited_paragraphs)
                st.session_state.story_manager.add_version(edited_text, "Manually edited")
                st.success("Edited version saved!")
                st.rerun()
            
            st.subheader("Refinement Tools")
            feedback = st.text_input(
                "How should we improve this?",
                "Make the dialogue more natural and descriptive"
            )
            
            refine_col1, refine_col2 = st.columns(2)
            with refine_col1:
                if st.button("Apply Changes", use_container_width=True):
                    with st.spinner(f"Refining with {selected_display}..."):
                        try:
                            current_text = join_paragraphs(st.session_state.edited_paragraphs)
                            refined_text = refine_text(
                                original_text=current_text,
                                user_feedback=feedback,
                                model=st.session_state.selected_model,
                                system_prompt=st.session_state.system_prompt
                            )
                            st.session_state.story_manager.add_version(refined_text, feedback)
                            st.session_state.edited_paragraphs = split_into_paragraphs(refined_text)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Refinement failed: {str(e)}")
            
            with refine_col2:
                if st.button("Start New Story", type="secondary", use_container_width=True):
                    st.session_state.story_manager = StoryManager()
                    st.session_state.edited_paragraphs = []
                    st.rerun()

    with col2:
        if st.session_state.story_manager.versions:
            st.subheader("Quick Refinement Presets")
            
            if st.button("Improve Readability", use_container_width=True):
                with st.spinner("Adjusting readability..."):
                    try:
                        current_text = join_paragraphs(st.session_state.edited_paragraphs)
                        refined = refine_text(
                            current_text,
                            "Improve readability: simplify complex sentences, adjust pacing",
                            model=st.session_state.selected_model,
                            system_prompt=st.session_state.system_prompt
                        )
                        st.session_state.story_manager.add_version(refined, "Readability improved")
                        st.session_state.edited_paragraphs = split_into_paragraphs(refined)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            
            if st.button("Enhance Descriptions", use_container_width=True):
                with st.spinner("Adding vivid descriptions..."):
                    try:
                        current_text = join_paragraphs(st.session_state.edited_paragraphs)
                        refined = refine_text(
                            current_text,
                            "Enhance descriptions: add sensory details, vivid imagery",
                            model=st.session_state.selected_model,
                            system_prompt=st.session_state.system_prompt
                        )
                        st.session_state.story_manager.add_version(refined, "Descriptions enhanced")
                        st.session_state.edited_paragraphs = split_into_paragraphs(refined)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            
            if st.button("Shorten Text", use_container_width=True):
                with st.spinner("Making concise..."):
                    try:
                        current_text = join_paragraphs(st.session_state.edited_paragraphs)
                        refined = refine_text(
                            current_text,
                            "Make this more concise by 30% without losing key information",
                            model=st.session_state.selected_model,
                            system_prompt=st.session_state.system_prompt
                        )
                        st.session_state.story_manager.add_version(refined, "Text shortened")
                        st.session_state.edited_paragraphs = split_into_paragraphs(refined)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            st.subheader("Paragraph Tools")
            if st.button("Add New Paragraph", use_container_width=True):
                st.session_state.edited_paragraphs.append("")
                st.session_state.editing_paragraph = len(st.session_state.edited_paragraphs) - 1
                st.rerun()

            if st.button("Combine Selected Paragraphs", use_container_width=True):
                selected = st.multiselect(
                    "Select paragraphs to combine",
                    options=[f"Paragraph {i+1}" for i in range(len(st.session_state.edited_paragraphs))]
                )
                if selected:
                    indices = [int(p.split()[1])-1 for p in selected]
                    combined = "\n\n".join([st.session_state.edited_paragraphs[i] for i in sorted(indices)])
                    # Remove the combined paragraphs and insert the new one at the first position
                    for i in sorted(indices, reverse=True):
                        st.session_state.edited_paragraphs.pop(i)
                    st.session_state.edited_paragraphs.insert(min(indices), combined)
                    st.rerun()

if __name__ == "__main__":
    main()