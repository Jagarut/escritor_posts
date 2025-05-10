import os
import time
import streamlit as st
from prompt_lib import SYSTEM_PROMPTS, USER_PROMPTS
from styles_lib import STYLE_PRESETS
from story_manager import StoryManager
from utils import (generate_text, refine_text, split_into_paragraphs, 
                   join_paragraphs, delete_paragraph, regenerate_paragraph, 
                   apply_style, generate_pdf, generate_epub, find_sentence_boundary_before_index,
                   insert_empty_paragraph, move_paragraph_up, move_paragraph_down, save_work, load_work
)

# Available models with proper naming
MODELS = {
    "Mistral Small": "mistral-small-latest",
    "Mistral Medium": "mistral-medium-latest",
    "Mistral Large": "mistral-large-latest",
    "Mistral Nemo": "open-mistral-nemo",
    "OpenRouter Qwen3:32b": "or/qwen/qwen3-32b:free",
    "Groq Gemma2:9b": "groq/gemma2-9b-it",
    "Groq Llama3.1:8b": "groq/llama-3.1-8b-instant",
    "Groq Qwen-Qwq:32b": "groq/qwen-qwq-32b",
    "Groq Llama Scout": "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "Groq Llama Maverick": "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
    "Groq LLama Versatile:70b": "groq/llama-3.3-70b-versatile",
    "Cerebras llama3.1-8b": "cerebras/llama3.1-8b",
    "Cerebras llama3.3-70b": "cerebras/llama-3.3-70b",
    "Cerebras llama-4-scout-17b": "cerebras/llama-4-scout-17b-16e-instruct",
    "SambaNova DeepSeek-V3-0324": "samba/DeepSeek-V3-0324",
    "SambaNova Llama-3.1-8B-Instruct": "samba/Meta-Llama-3.1-8B-Instruct",
    "SambaNova Llama-3.2-3B-Instruct": "samba/Meta-Llama-3.2-3B-Instruct",
    "SambaNova Llama-3.3-70B-Instruct": "samba/Meta-Llama-3.3-70B-Instruct",
    "SambaNova Llama-3.1-405B-Instruct": "samba/Meta-Llama-3.1-405B-Instruct",
    "Ollama Mistral": "ollama/mistral",
    "Ollama Hermes 3:3b": "ollama/hermes3:3b",
    "Ollama Llama3.2": "ollama/llama3.2:3b",
    "LM Studio Llama3.2": "LMstudio/llama-3.2-3b-instruct-uncensored",
    "LM Studio dolphin3.0-llama3.2": "LMstudio/dolphin3.0-llama3.2-3b",
    "LM Studio dolphin3.0-qwen2.5": "LMstudio/dolphin3.0-qwen2.5-3b",
    "LM Studio dolphin-2.9.4-gemma2-2b": "LMstudio/dolphin-2.9.4-gemma2-2b",
    "LM Studio Hermes 3:8b": "LMstudio/hermes-3-llama-3.1-8b",
    "LM Studio nsfw-3b": "LMstudio/nsfw-3b",
}

# System prompt options
# SYSTEM_PROMPTS = SYSTEM_PROMPTS

def main():
    st.set_page_config(page_title="AI Story Writer", layout="wide")
    st.title("📖 AI Story Writer")
    st.write("Generate and refine stories using Cloud or Local AI models")

    # Initialize session state
    if "story_manager" not in st.session_state:
        st.session_state.story_manager = StoryManager()
        # st.session_state.versions = None
        st.session_state.selected_model = "mistral-small-latest"
        st.session_state.temperature = 0.7
        st.session_state.system_prompt = SYSTEM_PROMPTS["Obedient AI"]
        st.session_state.user_prompt = USER_PROMPTS["Empty Prompt"]
        st.session_state.editing_paragraph = None
        st.session_state.edited_paragraphs = []
    
    if "regenerating_paragraph" not in st.session_state:
        st.session_state.regenerating_paragraph = None
    if "last_ai_paragraph" not in st.session_state:
        st.session_state.last_ai_paragraph = None
    if 'splitting_paragraph' not in st.session_state:
        st.session_state.splitting_paragraph = None
    
    if 'split_preview' not in st.session_state:
        st.session_state.split_preview = {
            'active': False,
            'paragraph_idx': None,
            'position': None,
            'original_text': ""
        }
    if "new_pos" not in st.session_state:
        st.session_state.new_pos = None
        
    # Apply temporary stored values after rerun
    if 'temp_model' in st.session_state:
        st.session_state.selected_model = st.session_state.temp_model
        del st.session_state.temp_model
        
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        current_model = st.session_state.selected_model
        selected_display = st.selectbox(
            "AI Model",
            options=list(MODELS.keys()),
            index=list(MODELS.values()).index(current_model),  # sets the default selected model in the select box
            key="model_select"
        )
        st.session_state.selected_model = MODELS[selected_display]
        st.info(f"Selected Model: {st.session_state.selected_model}")
        
        # System prompt selection
        selected_prompt = st.selectbox(
            "System Prompt",
            options=list(SYSTEM_PROMPTS.keys()),
            index=list(SYSTEM_PROMPTS.values()).index(st.session_state.system_prompt),
            key="sys_prompt_select"
        )
        st.session_state.system_prompt = SYSTEM_PROMPTS[selected_prompt]
        st.info(f"System Prompt: {st.session_state.system_prompt[:50]}...")
        
        st.markdown("---")
        # Writing styles
        default_styles = ["Tolkien"]  # Default style
        if 'pending_styles' in st.session_state:
            # Use loaded styles
            pending_styles = st.session_state.pop('pending_styles')
            # Filter to ensure all styles exist in STYLE_PRESETS
            default_styles = [s for s in pending_styles if s in STYLE_PRESETS]
            if not default_styles:  # If none of the loaded styles are valid
                default_styles = ["Tolkien"]
                
        selected_styles = st.multiselect(
            "Writing Style",
            options=list(STYLE_PRESETS.keys()),
            default=default_styles,  # Default style
            key="style_presets"
        )
        # Live preview
        with st.expander("Style Preview"):
            st.caption("Current style mix:")
            for style in selected_styles:
                st.markdown(f"- {style}: *{STYLE_PRESETS[style][:50]}...*")
                
        st.markdown("---")
        # User prompt selection
        selected_prompt = st.selectbox(
            "Instructions(automatic AI paragraph regeneration)",
            options=list(USER_PROMPTS.keys()),
            index=list(USER_PROMPTS.values()).index(st.session_state.user_prompt),
            key="user_prompt_select"
        )
        st.session_state.user_prompt = USER_PROMPTS[selected_prompt]
        st.info(f"Prompt: {st.session_state.user_prompt[:70]}...")
        
        # Temperature control
        st.session_state.temperature = st.slider(
            "Creativity (temperature)",
            0.0, 1.0, st.session_state.temperature, 0.1,
            key="temp_slider"
        )
        
        # Model info
        if st.session_state.selected_model.startswith("ollama"):
            st.info("Using local Ollama model. Ensure the model is pulled and Ollama is running.")
        if st.session_state.selected_model.startswith("LMstudio"):
            st.info("Using local LMstudio model. Ensure the model is loaded and LMstudio is running.")
        
        # Version history
        st.header("Version History")
        if st.session_state.story_manager.versions:
            # Add a selectbox for version recovery
            version_options = [
                f"v{i+1}: {v['feedback'][:30]}..." 
                for i, v in enumerate(st.session_state.story_manager.get_version_history())
            ]
            selected_version = st.selectbox(
                "Recover version:",
                options=version_options,
                index=len(version_options)-1,  # Default to latest
                key="version_selector"
            )

            # Add restore button
            if st.button("Restore This Version", key="restore_version"):
                version_index = version_options.index(selected_version)
                version = st.session_state.story_manager.get_version(version_index)
                st.session_state.edited_paragraphs = split_into_paragraphs(version["text"])
                st.rerun()

            # Keep your existing expander view
            for i, version in enumerate(st.session_state.story_manager.get_version_history()):
                with st.expander(f"v{i+1}: {version['feedback'][:50]}..."):
                    st.caption(version["feedback"])
                    st.code(version["text"][:200] + ("..." if len(version["text"]) > 200 else ""))      
                    
                    
                    
         # Replace the save/load section (lines 172-211) with this improved version
        with st.expander("💾 Save/Load Work"):
            # Save Section
            st.subheader("Save Current Work")
            save_name = st.text_input("Save as:", "my_story")
            if st.button("💾 Save"):
                # Determine what to save based on UI state
                if st.session_state.edited_paragraphs:
                    # We have paragraphs to save
                    paragraphs_to_save = st.session_state.edited_paragraphs
                else:
                    # We're in the "Create New Story" UI, nothing to save yet
                    st.warning("No content to save. Please generate a story first.")
                    paragraphs_to_save = []
                    
                if paragraphs_to_save:
                    file_path = save_work({
                        'edited_paragraphs': paragraphs_to_save,
                        'selected_model': st.session_state.selected_model,
                        'style_presets': st.session_state.get('style_presets', []),
                        'system_prompt': st.session_state.system_prompt
                    }, f"DRAFTS/{save_name}.json")
                    st.success(f"Saved to {file_path}")

            # Load Section 
            st.subheader("Load Previous Work")
            uploaded_file = st.file_uploader("Choose JSON file", type="json")

            if uploaded_file:
                st.write("File selected. Click Load to proceed.")
                if st.button("🔃 Load", key="load_button_unique"):
                    try:
                        st.write("Loading file...")  # Debug message
                        loaded = load_work(uploaded_file)

                        # Update paragraphs directly
                        if 'paragraphs' in loaded and loaded['paragraphs']:
                            # Store the paragraphs in session state
                            st.session_state.edited_paragraphs = loaded['paragraphs']

                            # Create a full text version from the paragraphs
                            full_text = join_paragraphs(loaded['paragraphs'])

                            # CRITICAL: Add this to the story manager to switch UI state
                            # This ensures we see the "Current Draft" UI with our paragraphs
                            st.session_state.story_manager.add_version(full_text, "Loaded from file")

                        # Update model if available
                        if loaded.get('model'):
                            st.session_state.selected_model = loaded.get('model')

                        # Update system prompt if available
                        if loaded.get('system_prompt'):
                            st.session_state.system_prompt = loaded.get('system_prompt')

                        # For style presets, we'll set a flag to update after rerun
                        if loaded.get('styles'):
                            st.session_state.pending_styles = loaded.get('styles')

                        st.success("Loaded successfully! Refreshing UI...")
                        time.sleep(1)  # Short delay to ensure the success message is seen
                        # Force a rerun to update the UI
                        st.rerun()
                    except Exception as e:
                        st.error(f"Load error: {str(e)}")
                        
            else:
                st.write("Please select a file to load.")


    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        # First draft generation
        if not st.session_state.story_manager.versions:
            st.subheader("Create New Story")
            user_prompt = st.text_area(
                "Story prompt",
                st.session_state.user_prompt,
                height=450
            )
            
            if st.button("Generate First Draft", use_container_width=True):
                with st.spinner(f"Generating with {selected_display}..."):
                    enhanced_system_prompt = apply_style(
                        st.session_state.get("style_presets", None),
                        st.session_state.system_prompt
                    )
                    try:
                        draft = generate_text(
                            prompt=f"{user_prompt}",
                            model=st.session_state.selected_model,
                            temperature=st.session_state.temperature,
                            system_prompt=enhanced_system_prompt
                        )
                        st.session_state.story_manager.add_version(draft, "Initial draft")
                        st.session_state.edited_paragraphs = split_into_paragraphs(draft)
                        st.rerun()  # Rerun to update UI
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
                            height=300
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
                        text_col, btn_col1, btn_col2 = st.columns([4, 0.5, 0.5])
                        with text_col:
                            # Add paragraph number as a label
                            st.markdown(f"**Paragraph {i+1}:**")
                            st.write(paragraph)
                        with btn_col1:
                            if st.button("⬆️", key=f"move_up_{i}", help="Move paragraph up", disabled=(i == 0)):
                               st.session_state.edited_paragraphs = move_paragraph_up(
                                   st.session_state.edited_paragraphs, i
                               )
                               st.rerun()
                        with btn_col2:       
                            if st.button("⬇️", key=f"move_down_{i}", help="Move paragraph down", disabled=(i == len(st.session_state.edited_paragraphs)-1)):
                                st.session_state.edited_paragraphs = move_paragraph_down(
                                    st.session_state.edited_paragraphs, i
                                )
                                st.rerun()   
                        
                        with btn_col1:        
                            if st.button("🔄", key=f"regenerate_{i}", help="Regenerate paragraph with AI"):
                                # Save current paragraph before regenerating
                                st.session_state.last_ai_paragraph = {
                                    'index': i,
                                    'content': st.session_state.edited_paragraphs[i]
                                }
                                with st.spinner(f"Regenerating paragraph {i+1}... "):
                                    st.session_state.regenerating_paragraph = i
                                    st.rerun()
                        with btn_col2:
                           if (st.session_state.last_ai_paragraph and 
                                st.session_state.last_ai_paragraph['index'] == i):

                                if st.button("🔥", key=f"undo_ai_{i}", help="Undo Regeneration"):
                                    st.session_state.edited_paragraphs[i] = st.session_state.last_ai_paragraph['content']
                                    st.session_state.last_ai_paragraph = None
                                    st.rerun()
                                    
                        with btn_col2:        
                            if st.button("✏️", key=f"edit_btn_{i}", help="Edit paragraph"):
                                st.session_state.editing_paragraph = i
                                st.rerun()
                        with btn_col1:
                            # Add this button to add a paragraph below
                            if st.button("➕", key=f"add_below_{i}", help="Add a paragraph below"):
                                st.session_state.edited_paragraphs = insert_empty_paragraph(
                                    st.session_state.edited_paragraphs, i
                                )
                                st.rerun()
                        with btn_col2:
                            # Add this delete button
                            if st.button("🗑️", key=f"delete_{i}", help="Delete Paragraph"):
                                st.session_state.edited_paragraphs = delete_paragraph(st.session_state.edited_paragraphs, i)
                                st.rerun()
                        with btn_col1:    
                            if len(st.session_state.story_manager.versions) > 1:
                                if st.button("↩️", type="secondary", help="Revert to the previous story", key=f"back_{i}"):
                                    try:
                                        index = len(st.session_state.story_manager.versions) - 1
                                        prev_version = st.session_state.story_manager.get_version(index)  # -2 gets previous version
                                        print(f"len(st.session_state.story_manager.versions): {len(st.session_state.story_manager.versions)}")
                                        print(f"Previous version: {prev_version}")
                                        if prev_version and "text" in prev_version:
                                            st.session_state.edited_paragraphs = split_into_paragraphs(prev_version["text"])
                                            st.success(f"Restored previous version from: {prev_version['feedback'][:50]}...")
                                            st.rerun()
                                        else:
                                            st.warning("Previous version not found or invalid")
                                    except Exception as e:
                                        st.error(f"Restore failed: {str(e)}")
                        
                        with btn_col2: 
                            if st.button("✂️", key=f"split_{i}", help="Split Paragraph"):
                                st.session_state.splitting_paragraph = i
                                st.rerun()
                        # Split interface (appears when activated)
                        if st.session_state.get('splitting_paragraph') == i:
                            # Initialize preview data if just activated
                            if not st.session_state.get('split_preview', {}).get('active', False):
                                # Find sentence boundary for initial split position
                                initial_split_pos = find_sentence_boundary_before_index(paragraph)
                                
                                st.session_state.split_preview = {
                                    'active': True,
                                    'paragraph_idx': i,
                                    'position': initial_split_pos,
                                    'original_text': paragraph
                                }
                            # Get current values
                            original = st.session_state.split_preview['original_text']    
                            current_pos = st.session_state.split_preview['position']
                            
                            # Initialize the slider value in session state if it doesn't exist
                            if f"split_slider_{i}" not in st.session_state:
                                st.session_state[f"split_slider_{i}"] = current_pos
    
                            # Define a callback function to update the position
                            def update_split_position():
                                if f"split_slider_{i}" in st.session_state:
                                    st.session_state.split_preview['position'] = st.session_state[f"split_slider_{i}"] 
                                    
                            # Add a button to snap to sentence boundary
                            if st.button("Snap to Sentence Boundary", key=f"snap_to_sentence_{i}", help="Snap to the index sentence boundary", use_container_width=True):   
                                sentence_boundary = find_sentence_boundary_before_index(original, st.session_state['new_pos'])
                                st.session_state.split_preview['position'] = sentence_boundary
                                st.session_state[f"split_slider_{i}"] = sentence_boundary
                                st.rerun()
                                
                            # Slider OUTSIDE the form for real-time updates
                            new_pos = st.slider(
                                "Adjust split point", 
                                0, len(original), 
                                key=f"split_slider_{i}",
                                on_change=update_split_position
                            )
                            
                            
                            st.session_state.new_pos = new_pos

                                
                            # Split the paragraph at the current position
                            part1 = original[:new_pos].strip()
                            part2 = original[new_pos:].strip()
                                
                            # Display real-time preview
                            # st.markdown("**Preview:**")
                            # col1par, col2par = st.columns(2)
                            # with col1par:
                            #     st.markdown("**First Part**")
                            #     st.text(part1)
                            # with col2par:
                            #     st.markdown("**Second Part**")
                            #     st.text(part2)    
                        
                            with st.form(key=f"split_form_{i}"):
                                # Form for confirmation only
                                st.text_area("First part:", value=part1, key=f"part1_{i}")
                                st.text_area("Second part:", value=part2, key=f"part2_{i}")
                                
                                confirm_col, can_col = st.columns(2)
                                with confirm_col:
                                    # Apply/cancel
                                    if st.form_submit_button("✓ Confirm Split"):
                                        # Only proceed if both parts have content
                                        if part1 and part2:
                                            st.session_state.edited_paragraphs[i:i+1] = [part1, part2]
                                            st.session_state.splitting_paragraph = None
                                            st.session_state.split_preview = {'active': False}  # Reset
                                            st.rerun()
                                        else:
                                            st.error("Cannot split: one of the parts would be empty.")
                                
                                with can_col:        
                                    if st.form_submit_button("✗ Cancel"):
                                        st.session_state.splitting_paragraph = None
                                        st.session_state.split_preview = {'active': False}
                                        st.rerun()

                        # AI Regeneration Input Box (only shows for the selected paragraph)
                        if st.session_state.get('regenerating_paragraph') == i:
                            # Context controls
                            with st.expander("🛠️ Context Controls", expanded=False):
                                context_window = st.slider(
                                    "Include surrounding paragraphs for context",
                                    0, 3, 0,  # Min: 0, Max: 3, Default: 1
                                    key=f"context_window_{i}"
                                )

                                # Visual context preview
                                if context_window > 0:
                                    st.caption(f"Preview (AI will see these {context_window*2} paragraphs):")

                                    # Show before paragraphs
                                    for ctx_i in range(max(0, i-context_window), i):
                                        st.text_area(
                                            f"↑ Previous paragraph {ctx_i+1}",
                                            st.session_state.edited_paragraphs[ctx_i],
                                            height=75,
                                            key=f"prev_{i}_{ctx_i}",
                                            disabled=True
                                        )

                                    # Current paragraph being edited
                                    st.text_area(
                                        "→ Current paragraph",
                                        paragraph,
                                        height=100,
                                        key=f"current_{i}",
                                        disabled=True
                                    )
                                    
                                    # Show after paragraphs
                                    for ctx_i in range(i+1, min(len(st.session_state.edited_paragraphs), i+1+context_window)):
                                        st.text_area(
                                            f"↓ Next paragraph {ctx_i+1}",
                                            st.session_state.edited_paragraphs[ctx_i],
                                            height=75,
                                            key=f"next_{i}_{ctx_i}",
                                            disabled=True
                                        )
                                
                            with st.form(key=f"regen_form_{i}"):
                                instruction = st.text_area(
                                    "How should we regenerate this paragraph?",
                                    # "Make this more descriptive and engaging",
                                    st.session_state.user_prompt,
                                    height=300,
                                    key=f"regen_instruction_{i}"
                                )

                                submit_col, cancel_col = st.columns(2)
                                with submit_col:
                                    if st.form_submit_button("Regenerate"):
                                        with st.spinner(f"Regenerating paragraph {i+1}..."):
                                            enhanced_system_prompt = apply_style(
                                                st.session_state.get("style_presets", None),
                                                st.session_state.system_prompt
                                            )
                                            try:
                                                regenerated = regenerate_paragraph(
                                                    paragraph=paragraph,
                                                    all_paragraphs=st.session_state.edited_paragraphs,
                                                    paragraph_index=i,
                                                    instruction=instruction,
                                                    context_window=context_window,  # Use the slider value
                                                    model=st.session_state.selected_model,
                                                    system_prompt=enhanced_system_prompt,
                                                    temperature=st.session_state.temperature
                                                )
                                                st.session_state.edited_paragraphs[i] = regenerated
                                                st.session_state.regenerating_paragraph = None  # Closes the edit window after regeneration
                                                st.rerun()
                                                st.success("Paragraph updated!")
                                            except Exception as e:
                                                st.error(f"Regeneration failed: {str(e)}")
                                                
                                with cancel_col:
                                    if st.form_submit_button("Cancel"):
                                        st.session_state.regenerating_paragraph = None
                                        st.rerun()
                                        
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
                "Add descriptive dialogue, expand on characters, or add more detail.",
            )
            
            refine_col1, refine_col2 = st.columns(2)
            with refine_col1:
                if st.button("Apply Changes", use_container_width=True):
                    with st.spinner(f"Refining with {selected_display}..."):
                        enhanced_system_prompt = apply_style(
                            st.session_state.get("style_presets", None),
                            st.session_state.system_prompt
                        )
                        try:
                            current_text = join_paragraphs(st.session_state.edited_paragraphs)
                            refined_text = refine_text(
                                original_text=current_text,
                                user_feedback=feedback,
                                model=st.session_state.selected_model,
                                system_prompt=enhanced_system_prompt
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
            
            if st.button("Eliminate repetition", use_container_width=True):
                with st.spinner("Eliminating repetition..."):
                    try:
                        current_text = join_paragraphs(st.session_state.edited_paragraphs)
                        refined = refine_text(
                            current_text,
                            user_feedback="""Revise the following story to eliminate repetitive phrases, 
                            tighten the prose, and enhance readability. 
                            Keep the original meaning and tone intact, but make the narrative more fluid and pleasant to read. 
                            Focus on varying sentence structure and word choice to avoid redundancy.""",
                            model=st.session_state.selected_model,
                            system_prompt=st.session_state.system_prompt
                        )
                        st.session_state.story_manager.add_version(refined, "Repetition elimination")
                        st.session_state.edited_paragraphs = split_into_paragraphs(refined)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            
            if st.button("Translate to Spanish", use_container_width=True):
                with st.spinner("Translating to Spanish..."):
                    try:
                        current_text = join_paragraphs(st.session_state.edited_paragraphs)
                        refined = refine_text(
                            current_text,
                            SYSTEM_PROMPTS["Traductor Sys Prompt"],
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

                        
            st.subheader("Export Options")

            if st.session_state.edited_paragraphs:
                current_story = join_paragraphs(st.session_state.edited_paragraphs)
            else:
                st.warning("No content to export. Please generate or edit a story first.")

            # PDF Settings - moved outside the button click handler
            with st.expander("PDF Settings", expanded=False):
                title = st.text_input("Title", "AI-Generated Story")
                author = st.text_input("Author", "OMG")

            # PDF Export
            if st.button("📄 Export to PDF", key="export_pdf", use_container_width=True):
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
                

            # EPUB Export
            with st.expander("EPUB Settings"):
                epub_title = st.text_input("Title", "AI-Generated Story", key="epub_title")
                epub_author = st.text_input("Author", "OMG", key="epub_author")
                # epub_filename = st.text_input("Filename", "ai_story.epub", key="epub_filename")

            if st.button("📚 Export to EPUB", key="export_epub", use_container_width=True):
                with st.spinner("Generating EPUB..."):
                    try:
                        epub_bytes = generate_epub(
                            current_story,
                            title=epub_title,
                            author=epub_author
                        )

                        st.download_button(
                            label="⬇️ Download EPUB",
                            data=epub_bytes,
                            file_name=f"{epub_title}.epub",
                            mime="application/epub+zip",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"EPUB generation failed: {str(e)}")
                        st.error("Please ensure all paragraphs contain valid content")
             
             

if __name__ == "__main__":
    main()