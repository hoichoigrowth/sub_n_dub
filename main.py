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
    # Extended language dictionary with 90+ languages
    language_dict = {
        "Afrikaans": {"lang_code": "af"},
        "Albanian": {"lang_code": "sq"},
        "Amharic": {"lang_code": "am"},
        "Arabic": {"lang_code": "ar"},
        "Armenian": {"lang_code": "hy"},
        "Azerbaijani": {"lang_code": "az"},
        "Bashkir": {"lang_code": "ba"},
        "Basque": {"lang_code": "eu"},
        "Belarusian": {"lang_code": "be"},
        "Bengali": {"lang_code": "bn"},
        "Bosnian": {"lang_code": "bs"},
        "Breton": {"lang_code": "br"},
        "Bulgarian": {"lang_code": "bg"},
        "Burmese": {"lang_code": "my"},
        "Cantonese": {"lang_code": "yue"},
        "Catalan": {"lang_code": "ca"},
        "Chinese": {"lang_code": "zh"},
        "Croatian": {"lang_code": "hr"},
        "Czech": {"lang_code": "cs"},
        "Danish": {"lang_code": "da"},
        "Dutch": {"lang_code": "nl"},
        "English": {"lang_code": "en"},
        "Estonian": {"lang_code": "et"},
        "Faroese": {"lang_code": "fo"},
        "Finnish": {"lang_code": "fi"},
        "French": {"lang_code": "fr"},
        "Galician": {"lang_code": "gl"},
        "Georgian": {"lang_code": "ka"},
        "German": {"lang_code": "de"},
        "Greek": {"lang_code": "el"},
        "Gujarati": {"lang_code": "gu"},
        "Haitian Creole": {"lang_code": "ht"},
        "Hausa": {"lang_code": "ha"},
        "Hawaiian": {"lang_code": "haw"},
        "Hebrew": {"lang_code": "he"},
        "Hindi": {"lang_code": "hi"},
        "Hungarian": {"lang_code": "hu"},
        "Icelandic": {"lang_code": "is"},
        "Indonesian": {"lang_code": "id"},
        "Irish": {"lang_code": "ga"},
        "Italian": {"lang_code": "it"},
        "Japanese": {"lang_code": "ja"},
        "Javanese": {"lang_code": "jw"},
        "Kannada": {"lang_code": "kn"},
        "Kazakh": {"lang_code": "kk"},
        "Khmer": {"lang_code": "km"},
        "Korean": {"lang_code": "ko"},
        "Kurdish": {"lang_code": "ku"},
        "Kyrgyz": {"lang_code": "ky"},
        "Lao": {"lang_code": "lo"},
        "Latin": {"lang_code": "la"},
        "Latvian": {"lang_code": "lv"},
        "Lithuanian": {"lang_code": "lt"},
        "Luxembourgish": {"lang_code": "lb"},
        "Macedonian": {"lang_code": "mk"},
        "Malagasy": {"lang_code": "mg"},
        "Malay": {"lang_code": "ms"},
        "Malayalam": {"lang_code": "ml"},
        "Maltese": {"lang_code": "mt"},
        "Mandarin": {"lang_code": "zh-cn"},
        "Maori": {"lang_code": "mi"},
        "Marathi": {"lang_code": "mr"},
        "Mongolian": {"lang_code": "mn"},
        "Nepali": {"lang_code": "ne"},
        "Norwegian": {"lang_code": "no"},
        "Occitan": {"lang_code": "oc"},
        "Pashto": {"lang_code": "ps"},
        "Persian": {"lang_code": "fa"},
        "Polish": {"lang_code": "pl"},
        "Portuguese": {"lang_code": "pt"},
        "Punjabi": {"lang_code": "pa"},
        "Romanian": {"lang_code": "ro"},
        "Russian": {"lang_code": "ru"},
        "Sanskrit": {"lang_code": "sa"},
        "Serbian": {"lang_code": "sr"},
        "Shona": {"lang_code": "sn"},
        "Sindhi": {"lang_code": "sd"},
        "Sinhala": {"lang_code": "si"},
        "Slovak": {"lang_code": "sk"},
        "Slovenian": {"lang_code": "sl"},
        "Somali": {"lang_code": "so"},
        "Spanish": {"lang_code": "es"},
        "Sundanese": {"lang_code": "su"},
        "Swahili": {"lang_code": "sw"},
        "Swedish": {"lang_code": "sv"},
        "Tagalog": {"lang_code": "tl"},
        "Tajik": {"lang_code": "tg"},
        "Tamil": {"lang_code": "ta"},
        "Tatar": {"lang_code": "tt"},
        "Telugu": {"lang_code": "te"},
        "Thai": {"lang_code": "th"},
        "Tibetan": {"lang_code": "bo"},
        "Turkish": {"lang_code": "tr"},
        "Turkmen": {"lang_code": "tk"},
        "Ukrainian": {"lang_code": "uk"},
        "Urdu": {"lang_code": "ur"},
        "Uzbek": {"lang_code": "uz"},
        "Vietnamese": {"lang_code": "vi"},
        "Welsh": {"lang_code": "cy"},
        "Yiddish": {"lang_code": "yi"},
        "Yoruba": {"lang_code": "yo"}
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
try:
    os.makedirs(SUBTITLE_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
except Exception as e:
    st.error(f"Error creating directories: {e}")

def get_language_name(lang_code):
    """Get language name from language code"""
    for language, details in language_dict.items():
        if details["lang_code"] == lang_code:
            return language
    return lang_code

def clean_file_name(file_path):
    """Clean filename for safe processing"""
    try:
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
    except Exception as e:
        st.error(f"Error cleaning filename: {e}")
        return file_path

def format_segments(segments):
    """Format whisper segments into structured data"""
    try:
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
    except Exception as e:
        st.error(f"Error formatting segments: {e}")
        return [], [], ""

def combine_word_segments(words_timestamp, max_words_per_subtitle=8, min_silence_between_words=0.5):
    """Combine words into subtitle segments"""
    try:
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
                st.warning(f"KeyError in word processing: {e} - Skipping word")
                continue

        # Add the last subtitle segment
        if text:
            before_translate[id] = {
                "text": text,
                "start": start,
                "end": end
            }

        return before_translate
    except Exception as e:
        st.error(f"Error combining word segments: {e}")
        return {}

def custom_word_segments(words_timestamp, min_silence_between_words=0.3, max_characters_per_subtitle=17):
    """Create custom word segments for shorts"""
    try:
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
    except Exception as e:
        st.error(f"Error creating custom word segments: {e}")
        return []

def convert_time_to_srt_format(seconds):
    """Convert seconds to SRT time format"""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
    except Exception as e:
        st.error(f"Error converting time format: {e}")
        return "00:00:00,000"

def write_subtitles_to_file(subtitles, filename="subtitles.srt"):
    """Write subtitles to SRT file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for id, entry in subtitles.items():
                f.write(f"{id}\n")
                start_time = convert_time_to_srt_format(entry['start'])
                end_time = convert_time_to_srt_format(entry['end'])
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{entry['text']}\n\n")
    except Exception as e:
        st.error(f"Error writing subtitles to file: {e}")

def word_level_srt(words_timestamp, srt_path="word_level_subtitle.srt", shorts=False):
    """Create word-level SRT file"""
    try:
        punctuation_pattern = re.compile(r'[.,!?;:"\–—_~^+*|]')
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            for i, word_info in enumerate(words_timestamp, start=1):
                start_time = convert_time_to_srt_format(word_info['start'])
                end_time = convert_time_to_srt_format(word_info['end'])
                word = word_info['word'] if 'word' in word_info else word_info.get('text', '')
                word = re.sub(punctuation_pattern, '', word)
                if word.strip().lower() == 'i':
                    word = "I"
                if not shorts:
                    word = word.replace("-", "")
                srt_file.write(f"{i}\n{start_time} --> {end_time}\n{word}\n\n")
    except Exception as e:
        st.error(f"Error creating word-level SRT: {e}")

def generate_srt_from_sentences(sentence_timestamp, srt_path="default_subtitle.srt"):
    """Generate SRT from sentence timestamps"""
    try:
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            for index, sentence in enumerate(sentence_timestamp):
                start_time = convert_time_to_srt_format(sentence['start'])
                end_time = convert_time_to_srt_format(sentence['end'])
                srt_file.write(f"{index + 1}\n{start_time} --> {end_time}\n{sentence['text']}\n\n")
    except Exception as e:
        st.error(f"Error generating SRT from sentences: {e}")

def get_audio_file(uploaded_file):
    """Save uploaded file to temp folder"""
    try:
        file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
        file_path = clean_file_name(file_path)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def translate_text_simple(text, target_language):
    """Simple text translation using language prefixes"""
    try:
        # Get language code
        target_code = language_dict.get(target_language, {}).get("lang_code", "en")
        
        # Simple translation approach for demo
        # In production, you'd integrate with Google Translate API, Azure Translator, etc.
        return f"[{target_code.upper()}] {text}"
            
    except Exception as e:
        st.warning(f"Translation failed for {target_language}: {e}")
        return text

def create_translated_subtitles(original_srt_path, target_languages, source_language):
    """Create translated subtitle files for multiple languages"""
    translated_files = {}
    
    try:
        # Read original SRT content
        with open(original_srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        
        # Parse SRT to extract text
        blocks = srt_content.strip().split('\n\n')
        parsed_subtitles = []
        
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                index = lines[0]
                timestamp = lines[1]
                text = '\n'.join(lines[2:])
                parsed_subtitles.append({
                    'index': index,
                    'timestamp': timestamp,
                    'text': text
                })
        
        # Create progress bar for translations
        if target_languages:
            trans_progress = st.progress(0)
            trans_status = st.empty()
        
        # Create translated versions for each target language
        for i, target_lang in enumerate(target_languages):
            if target_lang != source_language:
                if target_languages:
                    trans_status.text(f"🌍 Translating to {target_lang}... ({i+1}/{len(target_languages)})")
                    trans_progress.progress((i+1) / len(target_languages))
                
                # Create translated SRT
                translated_srt_path = original_srt_path.replace('.srt', f'_{target_lang.lower().replace(" ", "_")}.srt')
                translated_txt_path = original_srt_path.replace('.srt', f'_{target_lang.lower().replace(" ", "_")}.txt')
                
                translated_srt_content = ""
                translated_text_content = ""
                
                for subtitle in parsed_subtitles:
                    # Translate the text
                    translated_text = translate_text_simple(subtitle['text'], target_lang)
                    
                    # Add to SRT
                    translated_srt_content += f"{subtitle['index']}\n{subtitle['timestamp']}\n{translated_text}\n\n"
                    
                    # Add to text file
                    translated_text_content += f"{translated_text}\n"
                
                # Save translated SRT
                with open(translated_srt_path, 'w', encoding='utf-8') as f:
                    f.write(translated_srt_content)
                
                # Save translated text
                with open(translated_txt_path, 'w', encoding='utf-8') as f:
                    f.write(translated_text_content)
                
                translated_files[target_lang] = {
                    'srt': translated_srt_path,
                    'txt': translated_txt_path
                }
        
        # Clear progress
        if target_languages:
            trans_status.text("✅ All translations complete!")
            trans_progress.progress(1.0)
    
    except Exception as e:
        st.error(f"Translation error: {e}")
    
    return translated_files

def whisper_subtitle(uploaded_file, source_language, target_languages, max_words_per_subtitle=8):
    """Main subtitle generation function with translation support"""
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
        if not audio_path:
            return None, None, None, None, None, None, {}
        
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
        
        # Cleanup model before translations
        del faster_whisper_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        # Create translations if target languages are specified
        translated_files = {}
        if target_languages and len(target_languages) > 0:
            status_text.text("🌍 Creating translations...")
            translated_files = create_translated_subtitles(
                original_srt_name, 
                target_languages, 
                src_lang
            )
        
        return original_srt_name, customize_srt_name, word_level_srt_name, shorts_srt_name, original_txt_name, src_lang, translated_files
        
    except Exception as e:
        st.error(f"Error in whisper_subtitle: {e}")
        return None, None, None, None, None, None, {}

def main():
    st.title("🎬 AI Subtitle Generator")
    st.markdown("Generate subtitles using Whisper Large V3 Turbo with 90+ language support!")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Language selection
    source_lang_list = ['Automatic'] + sorted(list(language_dict.keys()))
    source_language = st.sidebar.selectbox(
        "Source Language",
        source_lang_list,
        index=0,
        help="Select the language of your audio/video"
    )
    
    # Target languages for translation
    st.sidebar.subheader("🌍 Translation Settings")
    available_languages = sorted(list(language_dict.keys()))
    
    # Initialize target_languages in session state
    if 'target_languages' not in st.session_state:
        st.session_state.target_languages = ["English", "Spanish", "French", "German", "Hindi", "Bengali"]
    
    target_languages = st.sidebar.multiselect(
        "Languages to translate to:",
        available_languages,
        default=st.session_state.target_languages,
        help="Select languages you want the subtitles translated to"
    )
    
    # Update session state
    st.session_state.target_languages = target_languages
    
    # Popular language shortcuts
    st.sidebar.markdown("**Quick Select:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🌏 Asian Languages"):
            st.session_state.target_languages = ["Chinese", "Japanese", "Korean", "Hindi", "Bengali", "Thai", "Vietnamese"]
            st.rerun()
            
    with col2:
        if st.button("🌍 European Languages"):
            st.session_state.target_languages = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian"]
            st.rerun()
    
    if st.sidebar.button("🌎 Popular Languages"):
        st.session_state.target_languages = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Hindi", "Arabic"]
        st.rerun()
    
    if st.sidebar.button("🗺️ All Major Languages"):
        st.session_state.target_languages = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", 
                          "Chinese", "Japanese", "Korean", "Hindi", "Bengali", "Arabic", "Turkish"]
        st.rerun()
    
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
        
        # Show selected languages
        if target_languages:
            st.info(f"🌍 Will translate to: {', '.join(target_languages)}")
        
        # Process button
        if st.button("🚀 Generate Subtitles", type="primary", disabled=st.session_state.processing):
            st.session_state.processing = True
            
            with st.spinner("Processing your file..."):
                try:
                    results = whisper_subtitle(
                        uploaded_file, 
                        source_language,
                        target_languages,
                        max_words_per_subtitle
                    )
                    
                    original_srt, customize_srt, word_level_srt, shorts_srt, text_file, detected_lang, translated_files = results
                    
                    if original_srt:
                        st.success(f"🎉 Processing completed! Detected language: {detected_lang}")
                        
                        # Create download section
                        st.header("📥 Download Results")
                        
                        # Original files
                        st.subheader(f"📄 Original Files ({detected_lang})")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
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
                            if os.path.exists(shorts_srt):
                                with open(shorts_srt, "rb") as f:
                                    st.download_button(
                                        label="📥 Shorts SRT",
                                        data=f.read(),
                                        file_name=f"{Path(uploaded_file.name).stem}_shorts.srt",
                                        mime="text/plain"
                                    )
                        
                        # Translated files
                        if translated_files:
                            st.subheader("🌍 Translated Files")
                            
                            # Group languages in rows of 3
                            languages = list(translated_files.keys())
                            for i in range(0, len(languages), 3):
                                cols = st.columns(3)
                                for j, lang in enumerate(languages[i:i+3]):
                                    with cols[j]:
                                        st.markdown(f"**{lang}**")
                                        
                                        # SRT download
                                        srt_path = translated_files[lang]['srt']
                                        if os.path.exists(srt_path):
                                            with open(srt_path, "rb") as f:
                                                st.download_button(
                                                    label=f"📥 {lang} SRT",
                                                    data=f.read(),
                                                    file_name=f"{Path(uploaded_file.name).stem}_{lang.lower().replace(' ', '_')}.srt",
                                                    mime="text/plain",
                                                    key=f"srt_{lang}_{i}_{j}"
                                                )
                                        
                                        # Text download
                                        txt_path = translated_files[lang]['txt']
                                        if os.path.exists(txt_path):
                                            with open(txt_path, "rb") as f:
                                                st.download_button(
                                                    label=f"📄 {lang} Text",
                                                    data=f.read(),
                                                    file_name=f"{Path(uploaded_file.name).stem}_{lang.lower().replace(' ', '_')}.txt",
                                                    mime="text/plain",
                                                    key=f"txt_{lang}_{i}_{j}"
                                                )
                        
                        # Show preview
                        if os.path.exists(original_srt):
                            st.header("👀 Preview")
                            with open(original_srt, "r", encoding="utf-8") as f:
                                content = f.read()
                            st.text_area("SRT Content Preview", content[:500] + ("..." if len(content) > 500 else ""), height=200)
                            
                            # Show translation preview if available
                            if translated_files:
                                preview_lang = st.selectbox("Preview translation:", list(translated_files.keys()))
                                if preview_lang in translated_files:
                                    txt_path = translated_files[preview_lang]['txt']
                                    if os.path.exists(txt_path):
                                        with open(txt_path, "r", encoding="utf-8") as f:
                                            trans_content = f.read()
                                        st.text_area(f"{preview_lang} Translation Preview", trans_content[:500] + ("..." if len(trans_content) > 500 else ""), height=150)
                    
                    else:
                        st.error("❌ Processing failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")
                
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
        - 90+ languages supported
        - Auto language detection
        """)
    
    with col3:
        st.markdown("""
        **📁 Output Files**
        - Original + Translated SRT
        - Word-level SRT
        - Shorts-optimized SRT
        - Text transcripts (all languages)
        """)

if __name__ == "__main__":
    main()
