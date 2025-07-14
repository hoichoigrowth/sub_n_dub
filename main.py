import streamlit as st
import os
import tempfile
import uuid
import re
import gc
import time
from pathlib import Path

# Try to import dependencies with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("PyTorch not available. Please install: pip install torch")

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    st.error("faster-whisper not available. Please install: pip install faster-whisper")

# Simple language dictionary (reduced for faster loading)
language_dict = {
    "English": {"lang_code": "en"},
    "Spanish": {"lang_code": "es"},
    "French": {"lang_code": "fr"},
    "German": {"lang_code": "de"},
    "Italian": {"lang_code": "it"},
    "Portuguese": {"lang_code": "pt"},
    "Russian": {"lang_code": "ru"},
    "Chinese": {"lang_code": "zh"},
    "Japanese": {"lang_code": "ja"},
    "Korean": {"lang_code": "ko"},
    "Hindi": {"lang_code": "hi"},
    "Arabic": {"lang_code": "ar"},
    "Bengali": {"lang_code": "bn"},
    "Dutch": {"lang_code": "nl"},
    "Turkish": {"lang_code": "tr"},
    "Polish": {"lang_code": "pl"},
    "Swedish": {"lang_code": "sv"},
    "Norwegian": {"lang_code": "no"},
    "Danish": {"lang_code": "da"},
    "Finnish": {"lang_code": "fi"}
}

# Streamlit configuration
st.set_page_config(
    page_title="🎬 AI Subtitle Generator",
    page_icon="🎬",
    layout="wide"
)

# Show dependency status
st.sidebar.header("📋 System Status")
st.sidebar.write(f"PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
st.sidebar.write(f"Whisper: {'✅' if WHISPER_AVAILABLE else '❌'}")

if TORCH_AVAILABLE:
    st.sidebar.write(f"CUDA: {'✅' if torch.cuda.is_available() else '❌'}")

# Only proceed if dependencies are available
if not (TORCH_AVAILABLE and WHISPER_AVAILABLE):
    st.error("Missing required dependencies. Please install them and restart the app.")
    st.code("""
pip install torch torchaudio
pip install faster-whisper
    """)
    st.stop()

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Setup folders
try:
    SUBTITLE_FOLDER = "generated_subtitles"
    TEMP_FOLDER = "temp_audio"
    
    os.makedirs(SUBTITLE_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
except Exception as e:
    st.error(f"Cannot create directories: {e}")
    st.stop()

def get_language_name(lang_code):
    """Get language name from language code"""
    for language, details in language_dict.items():
        if details["lang_code"] == lang_code:
            return language
    return "English"

def clean_filename(filename):
    """Clean filename for safe processing"""
    # Remove extension
    name = Path(filename).stem
    # Keep only alphanumeric characters and spaces
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    # Replace spaces with underscores
    clean_name = re.sub(r'\s+', '_', clean_name)
    # Add random suffix
    random_suffix = uuid.uuid4().hex[:8]
    return f"{clean_name}_{random_suffix}"

def convert_time_to_srt_format(seconds):
    """Convert seconds to SRT time format"""
    try:
        seconds = float(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
    except:
        return "00:00:00,000"

def create_srt_file(segments, output_path):
    """Create SRT file from segments"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = convert_time_to_srt_format(segment.start)
                end_time = convert_time_to_srt_format(segment.end)
                text = segment.text.strip()
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
        return True
    except Exception as e:
        st.error(f"Error creating SRT file: {e}")
        return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory"""
    try:
        clean_name = clean_filename(uploaded_file.name)
        file_extension = Path(uploaded_file.name).suffix
        temp_path = os.path.join(TEMP_FOLDER, f"{clean_name}{file_extension}")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def process_audio_file(file_path, source_language="auto"):
    """Process audio file with Whisper"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize model
        status_text.text("🧠 Loading Whisper model...")
        progress_bar.progress(20)
        
        # Use smaller model to avoid memory issues
        model_size = "base"  # Using base model instead of large for compatibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        status_text.text("🎯 Transcribing audio...")
        progress_bar.progress(50)
        
        # Transcribe
        if source_language == "auto":
            segments, info = model.transcribe(file_path)
            detected_language = get_language_name(info.language)
        else:
            lang_code = language_dict[source_language]["lang_code"]
            segments, info = model.transcribe(file_path, language=lang_code)
            detected_language = source_language
        
        status_text.text("📝 Creating subtitle files...")
        progress_bar.progress(80)
        
        # Convert segments to list for processing
        segments_list = list(segments)
        
        if not segments_list:
            status_text.text("❌ No speech detected")
            return None, None, None
        
        # Generate file paths
        base_name = clean_filename(Path(file_path).name)
        srt_path = os.path.join(SUBTITLE_FOLDER, f"{base_name}.srt")
        txt_path = os.path.join(SUBTITLE_FOLDER, f"{base_name}.txt")
        
        # Create SRT file
        create_srt_file(segments_list, srt_path)
        
        # Create text file
        with open(txt_path, 'w', encoding='utf-8') as f:
            for segment in segments_list:
                f.write(segment.text.strip() + " ")
        
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        return srt_path, txt_path, detected_language
        
    except Exception as e:
        status_text.text(f"❌ Error: {e}")
        st.error(f"Processing failed: {e}")
        return None, None, None

def main():
    st.title("🎬 Simple AI Subtitle Generator")
    st.markdown("Generate subtitles using Whisper AI")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    source_language = st.sidebar.selectbox(
        "Source Language",
        ["auto"] + list(language_dict.keys()),
        format_func=lambda x: "Automatic Detection" if x == "auto" else x,
        help="Select the language of your audio/video"
    )
    
    # File upload
    st.header("📁 Upload Audio/Video File")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['mp3', 'wav', 'm4a', 'mp4', 'avi', 'mov'],
        help="Supported formats: MP3, WAV, M4A, MP4, AVI, MOV"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📊 File size: {file_size:.1f} MB")
        
        if file_size > 50:
            st.warning("⚠️ Large files may take longer to process")
        
        # Process button
        if st.button("🚀 Generate Subtitles", type="primary", disabled=st.session_state.processing):
            st.session_state.processing = True
            
            try:
                # Save uploaded file
                temp_file_path = save_uploaded_file(uploaded_file)
                
                if temp_file_path:
                    # Process the file
                    srt_path, txt_path, detected_lang = process_audio_file(
                        temp_file_path, 
                        source_language if source_language != "auto" else "auto"
                    )
                    
                    # Cleanup temp file
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
                    
                    if srt_path and os.path.exists(srt_path):
                        st.success(f"🎉 Subtitles generated! Detected language: {detected_lang}")
                        
                        # Download section
                        st.header("📥 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # SRT download
                            with open(srt_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download SRT File",
                                    data=f.read(),
                                    file_name=f"{Path(uploaded_file.name).stem}.srt",
                                    mime="text/plain"
                                )
                        
                        with col2:
                            # Text download
                            if txt_path and os.path.exists(txt_path):
                                with open(txt_path, "rb") as f:
                                    st.download_button(
                                        label="📄 Download Text File",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}.txt",
                                        mime="text/plain"
                                    )
                        
                        # Preview
                        st.header("👀 Preview")
                        with open(srt_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.text_area(
                            "SRT Content (first 500 characters)", 
                            content[:500] + ("..." if len(content) > 500 else ""), 
                            height=200
                        )
                    else:
                        st.error("❌ Failed to generate subtitles")
                else:
                    st.error("❌ Failed to save uploaded file")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
            
            finally:
                st.session_state.processing = False
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎵 Supported Formats**
        - Audio: MP3, WAV, M4A
        - Video: MP4, AVI, MOV
        
        **🧠 AI Model**
        - OpenAI Whisper (Base Model)
        - Automatic language detection
        - High accuracy transcription
        """)
    
    with col2:
        st.markdown("""
        **📁 Output Files**
        - SRT subtitle file
        - Plain text transcript
        
        **💡 Tips**
        - Smaller files process faster
        - Clear audio gives better results
        - Supported languages: 20+ major languages
        """)

if __name__ == "__main__":
    main()
