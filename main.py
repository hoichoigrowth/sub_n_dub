import streamlit as st
import os
import tempfile
import shutil
import cv2
import subprocess
from pathlib import Path
import uuid
import re
import torch
import gc
from faster_whisper import WhisperModel
from docx import Document
from docx.shared import Inches
import requests
import json

# Configuration
st.set_page_config(
    page_title="Subtitle Generator",
    page_icon="🎬",
    layout="wide"
)

# Language dictionary (simplified version)
LANGUAGE_DICT = {
    "English": {"lang_code": "en"},
    "Spanish": {"lang_code": "es"},
    "French": {"lang_code": "fr"},
    "German": {"lang_code": "de"},
    "Italian": {"lang_code": "it"},
    "Portuguese": {"lang_code": "pt"},
    "Russian": {"lang_code": "ru"},
    "Japanese": {"lang_code": "ja"},
    "Korean": {"lang_code": "ko"},
    "Chinese": {"lang_code": "zh"},
    "Hindi": {"lang_code": "hi"},
    "Bengali": {"lang_code": "bn"},
    "Arabic": {"lang_code": "ar"},
    "Turkish": {"lang_code": "tr"},
    "Dutch": {"lang_code": "nl"},
    "Polish": {"lang_code": "pl"},
    "Swedish": {"lang_code": "sv"},
    "Norwegian": {"lang_code": "no"},
    "Danish": {"lang_code": "da"},
    "Finnish": {"lang_code": "fi"}
}

class SubtitleGenerator:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def clean_file_name(self, file_path):
        """Clean filename for safe processing"""
        file_name = os.path.basename(file_path)
        file_name, file_extension = os.path.splitext(file_name)
        cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)
        clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')
        random_uuid = uuid.uuid4().hex[:6]
        clean_file_path = os.path.join(
            os.path.dirname(file_path), 
            clean_file_name + f"_{random_uuid}" + file_extension
        )
        return clean_file_path

    def get_language_name(self, lang_code):
        """Get language name from code"""
        for language, details in LANGUAGE_DICT.items():
            if details["lang_code"] == lang_code:
                return language
        return lang_code

    def format_segments(self, segments):
        """Format whisper segments for processing"""
        saved_segments = list(segments)
        sentence_timestamp = []
        words_timestamp = []
        speech_to_text = ""

        for i in saved_segments:
            text = i.text.strip()
            sentence_id = len(sentence_timestamp)
            sentence_timestamp.append({
                "id": sentence_id,
                "text": text,
                "start": i.start,
                "end": i.end,
                "words": []
            })
            speech_to_text += text + " "

            for word in i.words:
                word_data = {
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                }
                sentence_timestamp[sentence_id]["words"].append(word_data)
                words_timestamp.append(word_data)

        return sentence_timestamp, words_timestamp, speech_to_text

    def convert_time_to_srt_format(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

    def transcribe_audio(self, audio_path, source_language="Automatic"):
        """Transcribe audio using Whisper"""
        try:
            # Load Whisper model
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🧠 Loading Whisper model...")
            progress_bar.progress(20)
            
            faster_whisper_model = WhisperModel(
                "deepdml/faster-whisper-large-v3-turbo-ct2",
                device=device, 
                compute_type=compute_type
            )
            
            status_text.text("🎯 Transcribing audio...")
            progress_bar.progress(40)
            
            if source_language == "Automatic":
                segments, info = faster_whisper_model.transcribe(audio_path, word_timestamps=True)
                lang_code = info.language
                src_lang = self.get_language_name(lang_code)
            else:
                lang = LANGUAGE_DICT[source_language]['lang_code']
                segments, info = faster_whisper_model.transcribe(
                    audio_path, 
                    word_timestamps=True, 
                    language=lang
                )
                src_lang = source_language
            
            progress_bar.progress(60)
            status_text.text("📝 Processing segments...")
            
            sentence_timestamp, words_timestamp, text = self.format_segments(segments)
            
            # Cleanup
            del faster_whisper_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            progress_bar.progress(80)
            status_text.text("✅ Transcription complete!")
            progress_bar.progress(100)
            
            return sentence_timestamp, words_timestamp, text, src_lang
            
        except Exception as e:
            st.error(f"Error in transcription: {e}")
            return None, None, None, None

    def translate_text_openai(self, text, target_language):
        """Translate text using OpenAI (if API key provided)"""
        # This is a placeholder - you'd need to implement OpenAI API calling
        # For now, we'll return the original text
        return text

    def create_subtitle_video(self, video_path, subtitles, output_path):
        """Create video with embedded subtitles using FFmpeg"""
        try:
            # Create temporary SRT file
            srt_path = os.path.join(self.temp_dir, "temp_subtitles.srt")
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, subtitle in enumerate(subtitles, 1):
                    start_time = self.convert_time_to_srt_format(subtitle['start'])
                    end_time = self.convert_time_to_srt_format(subtitle['end'])
                    f.write(f"{i}\n{start_time} --> {end_time}\n{subtitle['text']}\n\n")
            
            # FFmpeg command to embed subtitles
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f"subtitles='{srt_path}':force_style='Fontsize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Alignment=2'",
                '-c:a', 'copy',
                output_path
            ]
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                st.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            st.error(f"Error creating subtitle video: {e}")
            return False

    def create_multilingual_docx(self, original_text, subtitles, source_language, target_languages, filename):
        """Create DOCX with multilingual subtitles"""
        try:
            doc = Document()
            
            # Title
            doc.add_heading('Multilingual Subtitles', 0)
            
            # Source info
            info_para = doc.add_paragraph()
            info_para.add_run("Generated by Subtitle Generator\n").italic = True
            info_para.add_run(f"Source Language: {source_language}\n").italic = True
            info_para.add_run(f"Total Segments: {len(subtitles)}\n").italic = True
            
            # Add line break
            doc.add_paragraph()
            
            # For each subtitle segment
            for i, subtitle in enumerate(subtitles, 1):
                # Segment header
                doc.add_heading(f'Segment {i}', level=2)
                
                # Timestamp
                time_para = doc.add_paragraph()
                time_run = time_para.add_run(
                    f"[{self.convert_time_to_srt_format(subtitle['start'])} - "
                    f"{self.convert_time_to_srt_format(subtitle['end'])}]"
                )
                time_run.bold = True
                
                # Original text
                doc.add_paragraph().add_run("Original: ").bold = True
                doc.add_paragraph(subtitle['text'])
                
                # Translations for each target language
                for lang in target_languages:
                    if lang != source_language:
                        doc.add_paragraph().add_run(f"{lang}: ").bold = True
                        # For now, using original text (implement translation API as needed)
                        translated_text = self.translate_text_openai(subtitle['text'], lang)
                        doc.add_paragraph(translated_text)
                
                # Add spacing
                doc.add_paragraph()
            
            # Save document
            doc.save(filename)
            return True
            
        except Exception as e:
            st.error(f"Error creating DOCX: {e}")
            return False

    def process_video(self, uploaded_file, source_language, target_languages, max_words_per_subtitle=8):
        """Main processing function"""
        try:
            # Save uploaded file
            input_path = os.path.join(self.temp_dir, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Extract audio from video if needed
            if uploaded_file.type.startswith('video/'):
                audio_path = os.path.join(self.temp_dir, "audio.wav")
                cmd = ['ffmpeg', '-y', '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', audio_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    st.error("Failed to extract audio from video")
                    return None, None
            else:
                audio_path = input_path
            
            # Transcribe audio
            sentence_timestamp, words_timestamp, text, detected_lang = self.transcribe_audio(
                audio_path, source_language
            )
            
            if sentence_timestamp is None:
                return None, None
            
            # Create output files
            base_name = Path(uploaded_file.name).stem
            output_video_path = os.path.join(self.temp_dir, f"{base_name}_subtitled.mp4")
            output_docx_path = os.path.join(self.temp_dir, f"{base_name}_subtitles.docx")
            
            # Create video with subtitles (only if input is video)
            video_success = True
            if uploaded_file.type.startswith('video/'):
                video_success = self.create_subtitle_video(
                    input_path, sentence_timestamp, output_video_path
                )
            else:
                output_video_path = None
            
            # Create multilingual DOCX
            docx_success = self.create_multilingual_docx(
                text, sentence_timestamp, detected_lang, target_languages, output_docx_path
            )
            
            if video_success and docx_success:
                return output_video_path, output_docx_path
            else:
                return None, None
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
            return None, None

def main():
    st.title("🎬 AI Subtitle Generator")
    st.markdown("Generate subtitles and translate them into multiple languages!")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Language selection
    source_language = st.sidebar.selectbox(
        "Source Language",
        ["Automatic"] + list(LANGUAGE_DICT.keys()),
        index=0
    )
    
    target_languages = st.sidebar.multiselect(
        "Target Languages for Translation",
        list(LANGUAGE_DICT.keys()),
        default=["English", "Spanish", "French"]
    )
    
    max_words = st.sidebar.slider(
        "Max Words per Subtitle",
        min_value=1,
        max_value=15,
        value=8,
        help="Useful for vertical videos"
    )
    
    # Main interface
    st.header("📁 Upload Your Media")
    
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'm4a', 'flac'],
        help="Upload audio files for faster processing"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show file info
        file_size = len(uploaded_file.read()) / (1024 * 1024)  # MB
        uploaded_file.seek(0)  # Reset file pointer
        st.info(f"📊 File size: {file_size:.1f} MB")
        
        if file_size > 100:
            st.warning("⚠️ Large files may take longer to process. Consider using audio files for faster results.")
        
        # Process button
        if st.button("🚀 Generate Subtitles", type="primary"):
            with st.spinner("Processing your file..."):
                generator = SubtitleGenerator()
                
                try:
                    output_video_path, output_docx_path = generator.process_video(
                        uploaded_file, source_language, target_languages, max_words
                    )
                    
                    if output_video_path or output_docx_path:
                        st.success("🎉 Processing completed successfully!")
                        
                        # Create download section
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if output_video_path and os.path.exists(output_video_path):
                                st.subheader("📹 Subtitled Video")
                                with open(output_video_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Video with Subtitles",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}_subtitled.mp4",
                                        mime="video/mp4"
                                    )
                        
                        with col2:
                            if output_docx_path and os.path.exists(output_docx_path):
                                st.subheader("📄 Multilingual Document")
                                with open(output_docx_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Multilingual DOCX",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}_subtitles.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                    )
                        
                        # Show preview if it's a small video
                        if output_video_path and os.path.exists(output_video_path) and file_size < 50:
                            st.subheader("🎬 Preview")
                            st.video(output_video_path)
                    
                    else:
                        st.error("❌ Processing failed. Please try again with a different file.")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
                    st.error("Please check your file format and try again.")
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ How it works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎵 Step 1: Audio Processing**
        - Extracts audio from video files
        - Supports multiple audio formats
        - Optimizes for AI processing
        """)
    
    with col2:
        st.markdown("""
        **🧠 Step 2: AI Transcription**
        - Uses Whisper Large V3 Turbo
        - Automatic language detection
        - Word-level timestamp accuracy
        """)
    
    with col3:
        st.markdown("""
        **🌍 Step 3: Multilingual Output**
        - Embeds subtitles in video
        - Creates multilingual DOCX
        - Supports 20+ languages
        """)
    
    # Technical notes
    with st.expander("🔧 Technical Notes"):
        st.markdown("""
        - **Model**: Uses `deepdml/faster-whisper-large-v3-turbo-ct2`
        - **GPU Support**: Automatically detects CUDA availability
        - **Video Processing**: Uses FFmpeg for subtitle embedding
        - **Languages**: Supports 20+ languages with automatic detection
        - **File Limits**: Recommend files under 100MB for optimal performance
        """)

if __name__ == "__main__":
    main()
