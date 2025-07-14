import streamlit as st
import os
import tempfile
import shutil
import uuid
import re
import math
import torch
import gc
import time
from faster_whisper import WhisperModel
from pathlib import Path

# Import your language dictionary
try:
    from utils import language_dict
except ImportError:
    # Fallback language dictionary if utils.py is not available
    language_dict = {
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
        "Arabic": {"lang_code": "ar"}
    }

# Streamlit configuration
st.set_page_config(
    page_title="🎬 AI Subtitle Generator",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Setup folders
BASE_PATH = "."
SUBTITLE_FOLDER = f"{BASE_PATH}/generated_subtitle"
TEMP_FOLDER = f"{BASE_PATH}/subtitle_audio"

# Create folders if they don't exist
os.makedirs(SUBTITLE_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

def get_language_name(lang_code):
    """Get language name from language code"""
    for language, details in language_dict.items():
        if details["lang_code"] == lang_code:
            return language
    return lang_code

def clean_file_name(file_path):
    """Clean filename for safe processing"""
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)
    
    # Replace non-alphanumeric characters with underscore
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')
    
    # Generate random UUID for uniqueness
    random_uuid = uuid.uuid4().hex[:6]
    
    # Combine cleaned file name with original extension
    clean_file_path = os.path.join(
        os.path.dirname(file_path), 
        clean_file_name + f"_{random_uuid}" + file_extension
    )
    
    return clean_file_path

def format_segments(segments):
    """Format whisper segments into structured data"""
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

        # Process each word in the sentence
        if hasattr(i, 'words') and i.words:
            for word in i.words:
                word_data = {
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                }
                sentence_timestamp[sentence_id]["words"].append(word_data)
                words_timestamp.append(word_data)

    return sentence_timestamp, words_timestamp, speech_to_text

def combine_word_segments(words_timestamp, max_words_per_subtitle=8, min_silence_between_words=0.5):
    """Combine words into subtitle segments"""
    if max_words_per_subtitle <= 1:
        max_words_per_subtitle = 1
    
    before_translate = {}
    id = 1
    text = ""
    start = None
    end = None
    word_count = 0
    last_end_time = None

    for i in words_timestamp:
        try:
            word = i['word']
            word_start = i['start']
            word_end = i['end']

            # Check for sentence-ending punctuation
            is_end_of_sentence = word.endswith(('.', '?', '!'))

            # Check for conditions to create a new subtitle
            if ((last_end_time is not None and word_start - last_end_time > min_silence_between_words)
                or word_count >= max_words_per_subtitle
                or is_end_of_sentence):

                # Store the previous subtitle if there's any
                if text:
                    before_translate[id] = {
                        "text": text,
                        "start": start,
                        "end": end
                    }
                    id += 1

                # Reset for the new subtitle segment
                text = word
                start = word_start
                word_count = 1
            else:
                if word_count == 0:
                    start = word_start
                text += " " + word
                word_count += 1

            end = word_end
            last_end_time = word_end

        except KeyError as e:
            st.warning(f"KeyError: {e} - Skipping word")
            continue

    # Add the last subtitle segment
    if text:
        before_translate[id] = {
            "text": text,
            "start": start,
            "end": end
        }

    return before_translate

def custom_word_segments(words_timestamp, min_silence_between_words=0.3, max_characters_per_subtitle=17):
    """Create custom word segments for shorts"""
    before_translate = []
    text = ""
    start = None
    end = None

    i = 0
    while i < len(words_timestamp):
        word = words_timestamp[i]['word']
        word_start = words_timestamp[i]['start']
        word_end = words_timestamp[i]['end']

        # Handle hyphenated words
        if i + 1 < len(words_timestamp) and words_timestamp[i + 1]['word'].startswith("-"):
            combined_text = word + words_timestamp[i + 1]['word'][:]
            combined_start_time = word_start
            combined_end_time = words_timestamp[i + 1]['end']
            i += 1

            while i + 1 < len(words_timestamp) and words_timestamp[i + 1]['word'].startswith("-"):
                combined_text += words_timestamp[i + 1]['word'][:]
                combined_end_time = words_timestamp[i + 1]['end']
                i += 1
        else:
            combined_text = word
            combined_start_time = word_start
            combined_end_time = word_end

        # Check character limit
        if len(text) + len(combined_text) > max_characters_per_subtitle:
            if text:
                before_translate.append({
                    "word": text.strip(),
                    "start": start,
                    "end": end
                })
            text = combined_text
            start = combined_start_time
        else:
            if not text:
                start = combined_start_time
            text += " " + combined_text

        end = combined_end_time
        i += 1

    # Add final segment
    if text:
        before_translate.append({
            "word": text.strip(),
            "start": start,
            "end": end
        })

    return before_translate

def convert_time_to_srt_format(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def write_subtitles_to_file(subtitles, filename="subtitles.srt"):
    """Write subtitles to SRT file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for id, entry in subtitles.items():
            f.write(f"{id}\n")
            start_time = convert_time_to_srt_format(entry['start'])
            end_time = convert_time_to_srt_format(entry['end'])
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{entry['text']}\n\n")

def word_level_srt(words_timestamp, srt_path="word_level_subtitle.srt", shorts=False):
    """Create word-level SRT file"""
    punctuation_pattern = re.compile(r'[.,!?;:"\–—_~^+*|]')
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, word_info in enumerate(words_timestamp, start=1):
            start_time = convert_time_to_srt_format(word_info['start'])
            end_time = convert_time_to_srt_format(word_info['end'])
            word = word_info['word']
            word = re.sub(punctuation_pattern, '', word)
            if word.strip() == 'i':
                word = "I"
            if not shorts:
                word = word.replace("-", "")
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{word}\n\n")

def generate_srt_from_sentences(sentence_timestamp, srt_path="default_subtitle.srt"):
    """Generate SRT from sentence timestamps"""
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for index, sentence in enumerate(sentence_timestamp):
            start_time = convert_time_to_srt_format(sentence['start'])
            end_time = convert_time_to_srt_format(sentence['end'])
            srt_file.write(f"{index + 1}\n{start_time} --> {end_time}\n{sentence['text']}\n\n")

def get_audio_file(uploaded_file):
    """Save uploaded file to temp folder"""
    file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
    file_path = clean_file_name(file_path)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    return file_path

def whisper_subtitle(uploaded_file, source_language, max_words_per_subtitle=8):
    """Main subtitle generation function"""
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load model
        status_text.text("🧠 Loading Whisper model...")
        progress_bar.progress(10)
        
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu" 
            compute_type = "int8"
        
        faster_whisper_model = WhisperModel(
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            device=device, 
            compute_type=compute_type
        )
        
        status_text.text("📁 Processing audio file...")
        progress_bar.progress(20)
        
        audio_path = get_audio_file(uploaded_file)
        
        status_text.text("🎯 Transcribing audio...")
        progress_bar.progress(40)
        
        if source_language == "Automatic":
            segments, d = faster_whisper_model.transcribe(audio_path, word_timestamps=True)
            lang_code = d.language
            src_lang = get_language_name(lang_code)
        else:
            lang = language_dict[source_language]['lang_code']
            segments, d = faster_whisper_model.transcribe(
                audio_path, 
                word_timestamps=True, 
                language=lang
            )
            src_lang = source_language
        
        status_text.text("📝 Processing segments...")
        progress_bar.progress(60)
        
        sentence_timestamp, words_timestamp, text = format_segments(segments)
        
        # Cleanup audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Cleanup model
        del faster_whisper_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        status_text.text("🔧 Creating subtitle files...")
        progress_bar.progress(80)
        
        # Create different subtitle formats
        word_segments = combine_word_segments(
            words_timestamp, 
            max_words_per_subtitle=max_words_per_subtitle, 
            min_silence_between_words=0.5
        )
        shorts_segments = custom_word_segments(
            words_timestamp, 
            min_silence_between_words=0.3, 
            max_characters_per_subtitle=17
        )
        
        # Setup file names
        base_name = os.path.basename(uploaded_file.name).rsplit('.', 1)[0][:30]
        save_name = f"{SUBTITLE_FOLDER}/{base_name}_{src_lang}.srt"
        original_srt_name = clean_file_name(save_name)
        original_txt_name = original_srt_name.replace(".srt", ".txt")
        word_level_srt_name = original_srt_name.replace(".srt", "_word_level.srt")
        customize_srt_name = original_srt_name.replace(".srt", "_customize.srt")
        shorts_srt_name = original_srt_name.replace(".srt", "_shorts.srt")
        
        # Generate files
        generate_srt_from_sentences(sentence_timestamp, srt_path=original_srt_name)
        word_level_srt(words_timestamp, srt_path=word_level_srt_name)
        word_level_srt(shorts_segments, srt_path=shorts_srt_name, shorts=True)
        write_subtitles_to_file(word_segments, filename=customize_srt_name)
        
        with open(original_txt_name, 'w', encoding='utf-8') as f1:
            f1.write(text)
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        return original_srt_name, customize_srt_name, word_level_srt_name, shorts_srt_name, original_txt_name, src_lang
        
    except Exception as e:
        st.error(f"Error in whisper_subtitle: {e}")
        return None, None, None, None, None, None

def main():
    st.title("🎬 AI Subtitle Generator")
    st.markdown("Generate subtitles using Whisper Large V3 Turbo!")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Language selection
    source_lang_list = ['Automatic'] + list(language_dict.keys())
    source_language = st.sidebar.selectbox(
        "Source Language",
        source_lang_list,
        index=0
    )
    
    max_words_per_subtitle = st.sidebar.number_input(
        "Max Words per Subtitle Segment",
        min_value=1,
        max_value=20,
        value=8,
        help="Useful for vertical videos"
    )
    
    # File upload
    st.header("📁 Upload Your Media File")
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'm4a', 'flac', 'ogg'],
        help="Avoid uploading large video files. Audio files process faster."
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show file info
        file_size = len(uploaded_file.read()) / (1024 * 1024)
        uploaded_file.seek(0)  # Reset file pointer
        st.info(f"📊 File size: {file_size:.1f} MB")
        
        if file_size > 100:
            st.warning("⚠️ Large files may take longer to process.")
        
        # Process button
        if st.button("🚀 Generate Subtitles", type="primary", disabled=st.session_state.processing):
            st.session_state.processing = True
            
            with st.spinner("Processing your file..."):
                try:
                    results = whisper_subtitle(
                        uploaded_file, 
                        source_language, 
                        max_words_per_subtitle
                    )
                    
                    original_srt, customize_srt, word_level_srt, shorts_srt, text_file, detected_lang = results
                    
                    if original_srt:
                        st.success(f"🎉 Processing completed! Detected language: {detected_lang}")
                        
                        # Create download section
                        st.header("📥 Download Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("📄 Standard Files")
                            
                            if os.path.exists(original_srt):
                                with open(original_srt, "rb") as f:
                                    st.download_button(
                                        label="📥 Default SRT",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}_default.srt",
                                        mime="text/plain"
                                    )
                            
                            if os.path.exists(text_file):
                                with open(text_file, "rb") as f:
                                    st.download_button(
                                        label="📥 Text File",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}.txt",
                                        mime="text/plain"
                                    )
                        
                        with col2:
                            st.subheader("🎯 Custom Formats")
                            
                            if os.path.exists(customize_srt):
                                with open(customize_srt, "rb") as f:
                                    st.download_button(
                                        label="📥 Customized SRT",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}_custom.srt",
                                        mime="text/plain"
                                    )
                            
                            if os.path.exists(word_level_srt):
                                with open(word_level_srt, "rb") as f:
                                    st.download_button(
                                        label="📥 Word Level SRT",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}_words.srt",
                                        mime="text/plain"
                                    )
                        
                        with col3:
                            st.subheader("📱 For Shorts")
                            
                            if os.path.exists(shorts_srt):
                                with open(shorts_srt, "rb") as f:
                                    st.download_button(
                                        label="📥 Shorts SRT",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}_shorts.srt",
                                        mime="text/plain"
                                    )
                        
                        # Show preview
                        if os.path.exists(original_srt):
                            st.header("👀 Preview")
                            with open(original_srt, "r", encoding="utf-8") as f:
                                content = f.read()
                            st.text_area("SRT Content Preview", content[:500] + "...", height=200)
                    
                    else:
                        st.error("❌ Processing failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
                
                finally:
                    st.session_state.processing = False
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎵 Supported Formats**
        - Video: MP4, AVI, MOV, MKV
        - Audio: MP3, WAV, M4A, FLAC, OGG
        """)
    
    with col2:
        st.markdown("""
        **🧠 AI Model**
        - Whisper Large V3 Turbo
        - Word-level timestamps
        - 60+ languages supported
        """)
    
    with col3:
        st.markdown("""
        **📁 Output Files**
        - Standard SRT subtitles
        - Word-level SRT
        - Shorts-optimized SRT
        - Text transcript
        """)

if __name__ == "__main__":
    main()
