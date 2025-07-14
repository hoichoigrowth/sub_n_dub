import streamlit as st
import os
import tempfile
import subprocess
import uuid
import re
from pathlib import Path

# Simple language dictionary
LANGUAGES = {
    "Automatic": "auto",
    "English": "en",
    "Spanish": "es", 
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Hindi": "hi",
    "Bengali": "bn",
    "Arabic": "ar"
}

# App configuration
st.set_page_config(
    page_title="🎬 Subtitle Generator",
    page_icon="🎬",
    layout="wide"
)

def check_dependencies():
    """Check if required tools are available"""
    missing = []
    
    # Check FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("FFmpeg")
    
    return missing

def clean_filename(filename):
    """Clean filename for safe processing"""
    name = Path(filename).stem
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return f"{cleaned}_{uuid.uuid4().hex[:6]}"

def extract_audio_simple(video_path, audio_path):
    """Extract audio using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1',
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        st.error(f"Audio extraction failed: {e}")
        return False

def transcribe_with_whisper_cli(audio_path, language="auto"):
    """Use Whisper CLI for transcription"""
    try:
        # Use whisper command line tool
        cmd = ['whisper', audio_path, '--language', language, '--model', 'base', '--output_format', 'srt']
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(audio_path))
        
        if result.returncode == 0:
            # Find generated SRT file
            base_name = Path(audio_path).stem
            srt_path = os.path.join(os.path.dirname(audio_path), f"{base_name}.srt")
            if os.path.exists(srt_path):
                return srt_path
        
        return None
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

def create_subtitle_video_simple(video_path, srt_path, output_path):
    """Create video with subtitles using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"subtitles='{srt_path}':force_style='Fontsize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'",
            '-c:a', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        st.error(f"Video creation failed: {e}")
        return False

def create_simple_txt(srt_path, txt_path):
    """Convert SRT to simple text file"""
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract just the text from SRT
        lines = content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.isdigit() and '-->' not in line:
                text_lines.append(line)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))
        
        return True
    except Exception as e:
        st.error(f"Text file creation failed: {e}")
        return False

def main():
    st.title("🎬 Simple Subtitle Generator")
    st.markdown("Upload a video and get subtitles embedded!")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        st.error(f"❌ Missing dependencies: {', '.join(missing)}")
        st.markdown("""
        **To fix this:**
        1. Install FFmpeg: `sudo apt install ffmpeg`
        2. Install Whisper: `pip install openai-whisper`
        """)
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    source_language = st.sidebar.selectbox(
        "Source Language",
        list(LANGUAGES.keys()),
        index=0
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload your video file"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show file info
        file_size = len(uploaded_file.read()) / (1024 * 1024)
        uploaded_file.seek(0)
        st.info(f"📊 File size: {file_size:.1f} MB")
        
        if st.button("🚀 Generate Subtitles", type="primary"):
            with st.spinner("Processing your video..."):
                try:
                    # Create temp directory
                    temp_dir = tempfile.mkdtemp()
                    
                    # Save uploaded file
                    clean_name = clean_filename(uploaded_file.name)
                    video_path = os.path.join(temp_dir, f"{clean_name}.mp4")
                    
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    st.info("📹 Video saved, extracting audio...")
                    
                    # Extract audio
                    audio_path = os.path.join(temp_dir, f"{clean_name}.wav")
                    if not extract_audio_simple(video_path, audio_path):
                        st.error("Failed to extract audio")
                        return
                    
                    st.info("🎵 Audio extracted, transcribing...")
                    
                    # Transcribe
                    lang_code = LANGUAGES[source_language]
                    srt_path = transcribe_with_whisper_cli(audio_path, lang_code)
                    
                    if not srt_path:
                        st.error("Transcription failed")
                        return
                    
                    st.info("📝 Transcription complete, creating subtitle video...")
                    
                    # Create subtitle video
                    output_video_path = os.path.join(temp_dir, f"{clean_name}_subtitled.mp4")
                    if not create_subtitle_video_simple(video_path, srt_path, output_video_path):
                        st.error("Failed to create subtitle video")
                        return
                    
                    # Create text file
                    txt_path = os.path.join(temp_dir, f"{clean_name}_subtitles.txt")
                    create_simple_txt(srt_path, txt_path)
                    
                    st.success("🎉 Processing completed!")
                    
                    # Download buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if os.path.exists(output_video_path):
                            with open(output_video_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Video with Subtitles",
                                    data=f.read(),
                                    file_name=f"{Path(uploaded_file.name).stem}_subtitled.mp4",
                                    mime="video/mp4"
                                )
                    
                    with col2:
                        if os.path.exists(srt_path):
                            with open(srt_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    label="📥 Download SRT File",
                                    data=f.read(),
                                    file_name=f"{Path(uploaded_file.name).stem}.srt",
                                    mime="text/plain"
                                )
                    
                    with col3:
                        if os.path.exists(txt_path):
                            with open(txt_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    label="📥 Download Text File",
                                    data=f.read(),
                                    file_name=f"{Path(uploaded_file.name).stem}.txt",
                                    mime="text/plain"
                                )
                    
                    # Show preview for small files
                    if file_size < 20:
                        st.subheader("🎬 Preview")
                        if os.path.exists(output_video_path):
                            st.video(output_video_path)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Please try again with a different file.")
    
    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        **How to use:**
        1. Select source language (or use Automatic)
        2. Upload your video file (MP4, AVI, MOV, MKV)
        3. Click "Generate Subtitles"
        4. Download the results:
           - Video with embedded subtitles
           - SRT subtitle file
           - Plain text transcript
        
        **Requirements:**
        - FFmpeg must be installed
        - Whisper must be installed (`pip install openai-whisper`)
        - Video files under 50MB recommended for better performance
        """)

if __name__ == "__main__":
    main()
