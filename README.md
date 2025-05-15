# 🎓 Lecture Translator

Lecture Translator is a Python application built with Streamlit that automates the process of transcribing and translating video lectures. It uses OpenAI's Whisper model for  speech-to-text transcription and an OpenAI GPT model (configurable, default `gpt-4o-mini`) for translating the transcribed text into a specified target language. The application can output both SRT subtitle files (original and translated) and a new video file with the translated subtitles embedded.

## Features

-   **Video Upload:** Supports MP4 video file uploads.
-   **Accurate Transcription:** Utilizes various Whisper models (from `tiny` to `large-v3`) for transcription, allowing users to balance speed and accuracy.
-   **Machine Translation:** Leverages OpenAI's GPT models for translation into a wide range of languages.
-   **Customizable Translation Prompt:** Users can provide custom instructions to the translation model for more nuanced or domain-specific translations.
-   **Flexible Output Options:**
    -   Generate original language SRT subtitle file.
    -   Generate translated language SRT subtitle file.
    -   Generate a new MP4 video with translated subtitles embedded.
-   **Chunk Processing:** Splits longer videos into manageable chunks for more robust processing and to handle API limits.
-   **User-Friendly Interface:** Built with Streamlit for an intuitive web-based UI.
-   **Progress Tracking:** Real-time progress updates during processing.
-   **Error Handling:** Provides feedback on errors encountered during processing.
-   **Temporary File Management:** Cleans up temporary files after processing.
-   **Performance Metrics:** Displays processing times for different stages and token usage for OpenAI API calls.

## Requirements

-   Python 3.8+
-   FFmpeg: Must be installed and accessible in your system's PATH. FFmpeg is used by `moviepy` for video and audio manipulation.
-   ImageMagick (Optional, but recommended for subtitle embedding): `moviepy` may require ImageMagick for rendering text, especially with custom fonts or complex styles. The script includes a line `os.environ['IMAGEMAGICK_BINARY'] = "C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"` which you might need to adjust based on your ImageMagick installation path or remove if ImageMagick is in your PATH.
-   An OpenAI API Key for translation services.
-   CUDA-enabled GPU (Recommended for faster Whisper model processing, but CPU is also supported).

## Dependencies

The project relies on the following Python libraries:

-   `streamlit`
-   `torch` (and `torchaudio`, `torchvision` if installing PyTorch separately)
-   `openai`
-   `openai-whisper`
-   `moviepy`
-   `numpy` (often a dependency of other libraries)

You can install them using pip:

```bash
pip install streamlit torch openai openai-whisper moviepy numpy
```
or
```bash
pip install requirements.txt
```

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
    Or, simply save the Python script (e.g., `app.py`) to a directory on your computer.

2.  **Install FFmpeg:**
    * **Windows:** Download from [FFmpeg's official website](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.
    * **macOS (using Homebrew):** `brew install ffmpeg`
    * **Linux (using apt):** `sudo apt update && sudo apt install ffmpeg`

3.  **Install ImageMagick (Optional but Recommended):**
    * Download from [ImageMagick's official website](https://imagemagick.org/script/download.php).
    * Ensure it's added to your system's PATH or update the `IMAGEMAGICK_BINARY` environment variable in the script if needed.

4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    IMPORTANT: To use CUDA (using GPU instead of CPU for transcription) use a compatible CUDA torch build for example: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

## Usage

1.  **Set your OpenAI API Key:**
    The application requires an OpenAI API key for translation. You will be prompted to enter this in the UI.

2.  **Run the Streamlit application:**
    Open your terminal, navigate to the directory where you saved the script, and run:
    ```bash
    streamlit run your_script_name.py
    ```
    For example, if your script is named `app.py`:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

3.  **Using the Application:**
    * **Upload Video:** Click "Select MP4 video file" to upload your lecture video.
    * **Settings:**
        * Choose the **Transcription Model (Whisper)**. Larger models are more accurate but slower. `.en` models are optimized for English.
        * Enter the **Target Language for Translation** (e.g., "Spanish", "Estonian").
        * Select your desired **Output Options**:
            * `Generate Translated SRT file only`
            * `Generate Translated SRT and Video with embedded subtitles`
    * **Translation Tuning (Optional):**
        * Enter your **OpenAI API Key**.
        * Modify the **Custom Translation Instructions** if needed. The default prompt is designed for general subtitle translation.
    * **Start Processing:** Click the "🚀 Start Processing" button.
    * **Monitor Progress:** The UI will display progress messages and a progress bar.
    * **Download Results:** Once processing is complete, download links for the generated SRT files and/or the final video will appear. Processing times and token usage will also be displayed.

## Configuration

Most configurations are handled through the Streamlit UI. However, some constants are defined at the beginning of the script that you might want to adjust:

* `AVAILABLE_WHISPER_MODELS`: List of Whisper models available in the dropdown.
* `DEFAULT_MODEL`: The default Whisper model selected.
* `DEFAULT_LANGUAGE`: The default target language.
* `DEFAULT_TRANSLATION_PROMPT`: The default system prompt used for the translation API.
* `TRANSLATION_MODEL`: The OpenAI model used for translation (e.g., "gpt-4o-mini", "gpt-3.5-turbo").
* `CHUNK_DURATION`: Duration (in seconds) for splitting the audio into chunks (default is 120 seconds).
* **Subtitle Styling:**
    * `SUBTITLE_FONT`
    * `SUBTITLE_FONTSIZE`
    * `SUBTITLE_COLOR`
    * `SUBTITLE_BG_COLOR`
    * `SUBTITLE_POSITION`
* **Directories:**
    * `TEMP_DIR`: For temporary processing files.
    * `OUTPUT_DIR`: For final output files.
    * `CHUNK_AUDIO_DIR`: For audio chunks.
    * `CHUNK_SRT_DIR`: For SRT files generated per chunk.
* `IMAGEMAGICK_BINARY`: Path to your ImageMagick `magick.exe` (or equivalent). This is primarily for Windows if ImageMagick is not in PATH. On Linux/macOS, if ImageMagick is installed and in PATH, `moviepy` should find it automatically.

## File Structure (Generated)

When you run the application, it will create the following directories (if they don't exist):

* `temp_processing/`: Contains temporary files during processing.
    * `audio_chunks/`: Stores the individual audio chunks extracted from the video.
    * `*_full_audio.wav`: The full audio extracted from the input video.
* `output_processed/`: Contains the final output files.
    * `chunk_srts/`: Stores SRT files (original and translated) for each processed audio chunk.
    * `*_original_subtitles.srt`: The combined SRT file in the original language.
    * `*_<language>_subtitles.srt`: The combined SRT file in the target language.
    * `*_with_<language>subs.mp4`: The final video with embedded translated subtitles (if this option is selected).

## Error Handling and Logging

* The application provides user feedback for common errors directly in the Streamlit interface.
* More detailed error messages and tracebacks are printed to the console where the Streamlit application is running. This is useful for debugging.
* The script includes `try-except` blocks for robust error handling during file operations, API calls, and media processing.
* Warnings for malformed SRT entries or missing translations are printed to the console.
