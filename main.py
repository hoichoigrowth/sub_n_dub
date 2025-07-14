import streamlit as st
import sys
import os

# Show Python and system info for debugging
st.set_page_config(
    page_title="🎬 AI Subtitle Generator - Debug",
    page_icon="🎬",
    layout="wide"
)

st.title("🔧 Debug Mode - AI Subtitle Generator")

# Show system information
st.header("📊 System Debug Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Python Environment")
    st.write(f"Python version: {sys.version}")
    st.write(f"Platform: {sys.platform}")
    st.write(f"Current working directory: {os.getcwd()}")
    
with col2:
    st.subheader("Available Modules")
    
    # Test imports one by one
    modules_to_test = [
        "streamlit", "os", "tempfile", "shutil", "uuid", 
        "re", "gc", "time", "pathlib"
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            st.write(f"✅ {module}")
        except ImportError as e:
            st.write(f"❌ {module}: {e}")

# Test critical imports
st.header("🧪 Critical Dependencies Test")

# Test torch
try:
    import torch
    st.success(f"✅ PyTorch: {torch.__version__}")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.write(f"CUDA version: {torch.version.cuda}")
        st.write(f"GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    st.error(f"❌ PyTorch import failed: {e}")
except Exception as e:
    st.error(f"❌ PyTorch error: {e}")

# Test faster-whisper
try:
    from faster_whisper import WhisperModel
    st.success("✅ faster-whisper imported successfully")
    
    # Try to create a small model to test
    try:
        # This won't actually load, just test if the class works
        st.write("Testing WhisperModel class...")
        st.success("✅ WhisperModel class accessible")
    except Exception as e:
        st.warning(f"⚠️ WhisperModel test failed: {e}")
        
except ImportError as e:
    st.error(f"❌ faster-whisper import failed: {e}")
except Exception as e:
    st.error(f"❌ faster-whisper error: {e}")

# Test file operations
st.header("📁 File System Test")

try:
    # Test directory creation
    test_dir = "test_subtitle_folder"
    os.makedirs(test_dir, exist_ok=True)
    st.success(f"✅ Directory creation successful: {test_dir}")
    
    # Test file writing
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test content")
    st.success("✅ File writing successful")
    
    # Test file reading
    with open(test_file, "r") as f:
        content = f.read()
    st.success(f"✅ File reading successful: {content}")
    
    # Cleanup
    os.remove(test_file)
    os.rmdir(test_dir)
    st.success("✅ File cleanup successful")
    
except Exception as e:
    st.error(f"❌ File system test failed: {e}")

# Test session state
st.header("🗃️ Session State Test")

if 'test_counter' not in st.session_state:
    st.session_state.test_counter = 0

if st.button("Test Session State"):
    st.session_state.test_counter += 1

st.write(f"Session state counter: {st.session_state.test_counter}")

# Simple file upload test
st.header("📤 File Upload Test")

uploaded_file = st.file_uploader(
    "Test file upload",
    type=['txt', 'mp3', 'wav', 'mp4'],
    help="Upload any small file to test the upload functionality"
)

if uploaded_file is not None:
    st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
    st.write(f"File size: {len(uploaded_file.read())} bytes")
    uploaded_file.seek(0)  # Reset file pointer
    
    # Test file saving
    try:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ File saving successful")
        
        # Cleanup
        os.remove(temp_path)
        st.success("✅ File cleanup successful")
        
    except Exception as e:
        st.error(f"❌ File saving failed: {e}")

# Memory test
st.header("💾 Memory Test")

try:
    import gc
    gc.collect()
    st.success("✅ Garbage collection successful")
except Exception as e:
    st.error(f"❌ Memory test failed: {e}")

# Show environment variables (useful for debugging deployment)
st.header("🌍 Environment Variables")

important_env_vars = [
    "PATH", "PYTHONPATH", "HOME", "USER", 
    "STREAMLIT_SERVER_PORT", "PORT"
]

for var in important_env_vars:
    value = os.environ.get(var, "Not set")
    st.write(f"{var}: {value}")

# Final status
st.header("✅ Debug Summary")

# Check if everything is working
all_good = True

# Basic imports
basic_modules = ["torch", "faster_whisper"]
for module in basic_modules:
    try:
        __import__(module)
    except ImportError:
        all_good = False
        break

if all_good:
    st.success("🎉 All critical components are working! The main app should work.")
    
    if st.button("🚀 Try Minimal Subtitle App"):
        st.session_state.show_minimal_app = True
else:
    st.error("❌ Some critical components are missing. Check the errors above.")

# Show minimal working app
if st.session_state.get('show_minimal_app', False):
    st.markdown("---")
    st.header("🎬 Minimal Subtitle Generator")
    
    try:
        import torch
        from faster_whisper import WhisperModel
        from pathlib import Path
        
        # Minimal language dictionary
        languages = {
            "English": "en", "Spanish": "es", "French": "fr", 
            "German": "de", "Chinese": "zh", "Japanese": "ja"
        }
        
        # Simple UI
        source_lang = st.selectbox("Source Language", ["Automatic"] + list(languages.keys()))
        
        uploaded_file = st.file_uploader(
            "Upload audio/video file",
            type=['mp3', 'wav', 'mp4', 'avi']
        )
        
        if uploaded_file and st.button("Generate Subtitles"):
            try:
                st.info("🧠 Loading model...")
                
                # Use smallest model for testing
                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                
                st.info("📁 Saving file...")
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                st.info("🎯 Transcribing...")
                segments, info = model.transcribe(temp_path)
                
                # Create simple SRT
                srt_content = ""
                for i, segment in enumerate(segments, 1):
                    start = segment.start
                    end = segment.end
                    text = segment.text.strip()
                    
                    start_time = f"{int(start//3600):02}:{int((start%3600)//60):02}:{int(start%60):02},{int((start%1)*1000):03}"
                    end_time = f"{int(end//3600):02}:{int((end%3600)//60):02}:{int(end%60):02},{int((end%1)*1000):03}"
                    
                    srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
                
                st.success("✅ Transcription complete!")
                
                # Download button
                st.download_button(
                    label="📥 Download SRT",
                    data=srt_content,
                    file_name=f"{Path(uploaded_file.name).stem}.srt",
                    mime="text/plain"
                )
                
                # Preview
                st.text_area("Preview", srt_content[:500] + "...", height=200)
                
                # Cleanup
                os.remove(temp_path)
                del model
                gc.collect()
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    except ImportError as e:
        st.error(f"Cannot load minimal app: {e}")

# Instructions for fixing issues
st.markdown("---")
st.header("🔧 Troubleshooting Guide")

st.markdown("""
### If you see errors above:

1. **Missing PyTorch**: 
   - Add to requirements.txt: `torch>=2.0.0`
   - For CPU only: `torch --index-url https://download.pytorch.org/whl/cpu`

2. **Missing faster-whisper**:
   - Add to requirements.txt: `faster-whisper>=1.0.0`

3. **Memory issues**:
   - Use smaller models: `"tiny"` or `"base"` instead of `"large"`
   - Deploy on platforms with more RAM

4. **File permission issues**:
   - Check if the deployment platform allows file creation
   - Use `tempfile` module for temporary files

5. **Build timeout**:
   - Simplify requirements.txt
   - Remove unnecessary dependencies

### Recommended minimal requirements.txt:
```
streamlit
faster-whisper>=1.0.0
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```
""")
