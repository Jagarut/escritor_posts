import streamlit as st
from prompt_lib import PROMPT_LIBRARY
from story_manager import StoryManager
from utils import generate_text, refine_text, split_into_paragraphs, join_paragraphs, regenerate_paragraph, generate_pdf
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
                # Create a container for each paragraph
                with st.container():
                    # If in editing mode for this paragraph
                    if st.session_state.editing_paragraph == i:
                        edited = st.text_area(
                            f"Paragraph {i+1}",
                            paragraph,
                            key=f"edit_{i}",
                            height=150
                        )
                        
                        # Create top-level columns for Save/Cancel buttons
                        save_col, cancel_col = st.columns(2)
                        with save_col:
                            if st.button("Save", key=f"save_{i}"):
                                st.session_state.edited_paragraphs[i] = edited
                                st.session_state.editing_paragraph = None
                                st.rerun()
                        with cancel_col:
                            if st.button("Cancel", key=f"cancel_{i}"):
                                st.session_state.editing_paragraph = None
                                st.rerun()
                    else:
                        # Display paragraph and buttons side by side
                        text_col, btn_col = st.columns([4, 1])
                        with text_col:
                            # Add paragraph number as a label
                            st.markdown(f"**Paragraph {i+1}:**")
                            st.write(paragraph)
                        with btn_col:
                            if st.button("✏️ Edit", key=f"edit_btn_{i}"):
                                st.session_state.editing_paragraph = i
                                st.rerun()
                            if st.button("🔄 AI", key=f"regenerate_{i}"):
                                with st.spinner(f"Regenerating paragraph {i+1}..."):
                                    try:
                                        context = {
                                            "previous_paragraphs": st.session_state.edited_paragraphs[:i],
                                            "next_paragraphs": st.session_state.edited_paragraphs[i+1:]
                                        }
                                        regenerated = regenerate_paragraph(
                                            paragraph,
                                            context=context,
                                            model=st.session_state.selected_model,
                                            system_prompt=st.session_state.system_prompt,
                                            temperature=st.session_state.temperature
                                        )
                                        st.session_state.edited_paragraphs[i] = regenerated
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Regeneration failed: {str(e)}")
                    
                    # Add a visual separator between paragraphs
                    st.markdown("---")
            
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
            
            if st.button("Translate to Spanish", use_container_width=True):
                with st.spinner("Translating to Spanish..."):
                    try:
                        current_text = join_paragraphs(st.session_state.edited_paragraphs)
                        refined = refine_text(
                            current_text,
                            PROMPT_LIBRARY["Traductor English to Spanish"],
                            model=st.session_state.selected_model,
                            system_prompt=st.session_state.system_prompt
                        )
                        st.session_state.story_manager.add_version(refined, "Translated to Spanish")
                        st.session_state.edited_paragraphs = split_into_paragraphs(refined)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
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
        
            # Add New Paragraph button
            if st.button("Add New Paragraph", use_container_width=True):
                st.session_state.edited_paragraphs.append("")
                st.session_state.editing_paragraph = len(st.session_state.edited_paragraphs) - 1
                st.rerun()

            # Combine Paragraphs - Fixed Version
            st.write("---")
            st.subheader("Combine Paragraphs")

            # Create a multiselect with paragraph previews
            para_options = [
                f"Paragraph {i+1}: {para[:30]}..." if len(para) > 30 else f"Paragraph {i+1}: {para}"
                for i, para in enumerate(st.session_state.edited_paragraphs)
            ]

            selected = st.multiselect(
                "Select paragraphs to combine (in order):",
                options=para_options,
                format_func=lambda x: x
            )

            if st.button("Combine Selected Paragraphs", use_container_width=True):
                if len(selected) > 1:
                    try:
                        # Get the original indices from the selection
                        selected_indices = [para_options.index(sel) for sel in selected]

                        # Combine the paragraphs with double newlines between them
                        combined_text = '\n\n'.join(
                            st.session_state.edited_paragraphs[i] for i in selected_indices
                        )

                        # Remove the original paragraphs (starting from highest index first)
                        for i in sorted(selected_indices, reverse=True):
                            st.session_state.edited_paragraphs.pop(i)

                        # Insert the combined text at the first original position
                        st.session_state.edited_paragraphs.insert(
                            min(selected_indices), 
                            combined_text
                        )

                        st.rerun()
                    except Exception as e:
                        st.error(f"Error combining paragraphs: {str(e)}")
                else:
                    st.warning("Please select at least 2 paragraphs to combine")

            # Add AI regeneration with instructions and numbered paragraphs
            st.subheader("AI Paragraph Regeneration")
            regen_instruction = st.text_input(
                "Instructions for regeneration",
                "Make this more descriptive and engaging",
                key="regen_instruction"
            )
            
            # Create options with paragraph numbers and preview text
            paragraph_options = []
            for i, para in enumerate(st.session_state.edited_paragraphs):
                preview = para[:30] + "..." if len(para) > 30 else para
                paragraph_options.append(f"Para {i+1}: {preview}")
                
            selected_para = st.selectbox(
                "Select paragraph to regenerate",
                options=paragraph_options
            )
            
            if st.button("Regenerate Selected Paragraph", use_container_width=True):
                # Extract paragraph index from the selected option
                para_index = int(selected_para.split(":")[0].split()[1])-1
                with st.spinner(f"Regenerating paragraph {para_index+1}..."):
                    try:
                        context = {
                            "previous_paragraphs": st.session_state.edited_paragraphs[:para_index],
                            "next_paragraphs": st.session_state.edited_paragraphs[para_index+1:]
                        }
                        regenerated = regenerate_paragraph(
                            st.session_state.edited_paragraphs[para_index],
                            instruction=regen_instruction,
                            context=context,
                            model=st.session_state.selected_model,
                            system_prompt=st.session_state.system_prompt,
                            temperature=st.session_state.temperature
                        )
                        st.session_state.edited_paragraphs[para_index] = regenerated
                        st.rerun()
                    except Exception as e:
                        st.error(f"Regeneration failed: {str(e)}")       
                        
                        st.subheader("Export Options")
            
            # PDF Settings - moved outside the button click handler
            with st.expander("PDF Settings", expanded=False):
                title = st.text_input("Title", "My AI-Generated Story")
                author = st.text_input("Author", "AI Story Writer")
            
            # PDF Export
            if st.button("📄 Export to PDF", use_container_width=True):
                if st.session_state.edited_paragraphs:
                    current_story = join_paragraphs(st.session_state.edited_paragraphs)

                    # Generate PDF
                    try:
                        pdf = generate_pdf(current_story, title, author)
                        
                        # Create download link
                        from io import BytesIO
                        pdf_bytes = BytesIO()
                        pdf.output(pdf_bytes)
                        pdf_bytes.seek(0)
                        
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=f"{title.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
                        st.info("Make sure you have the fpdf library installed. Run: pip install fpdf")
                else:
                    st.warning("No content to export. Please generate or edit a story first.")
             

if __name__ == "__main__":
    main()