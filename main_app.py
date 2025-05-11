import streamlit as st
import os
import time
import shutil
import traceback
from threading import Thread
from queue import Queue
import numpy as np # Keep if needed by dependencies, though not directly used now

os.environ['IMAGEMAGICK_BINARY'] = "C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# --- Dependencies ---
import torch
from openai import OpenAI
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from moviepy.video.tools.subtitles import SubtitlesClip


# --- Application Configuration ---
# UI Defaults & Options
AVAILABLE_WHISPER_MODELS = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3']
DEFAULT_MODEL = 'medium.en'
DEFAULT_LANGUAGE = 'Estonian'
DEFAULT_TRANSLATION_PROMPT = "You are a professional translator specialized in subtitle translation. Translate the following subtitle text to {target_language}, preserving the nuanced meaning and context of the original text. Ensure the translation flows naturally while staying true to the original intent. Output ONLY the translated text."
TRANSLATION_MODEL = "gpt-4o-mini"

# Processing Parameters
CHUNK_DURATION = 120

# Subtitle Styling
SUBTITLE_FONT = 'Calibri'
SUBTITLE_FONTSIZE = 24
SUBTITLE_COLOR = 'white'
SUBTITLE_BG_COLOR = 'rgba(0, 0, 0, 0.5)'
SUBTITLE_POSITION = ('center', 'bottom')

# Directories
TEMP_DIR = "temp_processing"
OUTPUT_DIR = "output_processed"
CHUNK_AUDIO_DIR = os.path.join(TEMP_DIR, "audio_chunks")
CHUNK_SRT_DIR = os.path.join(OUTPUT_DIR, "chunk_srts")

# --- Helper Functions ---

def format_time(seconds):
    """Convert seconds to SRT time format HH:MM:SS,mmm"""
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def srt_time_to_seconds(time_str):
    """Convert SRT time string HH:MM:SS,mmm to seconds"""
    try:
        time_str = time_str.replace(',', '.')
        hours, mins, secs = map(float, time_str.split(':'))
        return hours * 3600 + mins * 60 + secs
    except ValueError:
        print(f"Error parsing SRT time: {time_str}")
        raise ValueError(f"Invalid SRT time format: {time_str}")

def safe_remove(path):
    """Safely remove a file or directory."""
    try:
        if path is None or not os.path.exists(path):
            return
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
            # print(f"Removed file: {path}")
    except OSError as e:
        print(f"Error removing {path}: {e}")

def extract_full_audio(video_path, output_dir, progress_queue=None):
    """Extract full audio, report progress."""
    if progress_queue: progress_queue.put("Extracting audio...")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(video_path)
    timestamp = str(int(time.time() * 1000))
    audio_filename = f"{os.path.splitext(base_name)[0]}_full_audio.wav"
    audio_path = os.path.join(output_dir, audio_filename)
    video = None
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            if progress_queue: progress_queue.put("❌ Error: No audio track found in video.")
            raise ValueError(f"No audio found in video file: {video_path}")

        video.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        video.close()
        if progress_queue: progress_queue.put("Audio extracted successfully.")
        return audio_path
    except Exception as e:
        safe_remove(audio_path)
        if video: video.close()
        if progress_queue: progress_queue.put(f"❌ Error extracting audio: {e}")
        raise Exception(f"Error extracting audio from {video_path}: {e}")

def split_audio_into_chunks(audio_path, chunk_duration, output_dir, progress_queue=None):
    """Split audio, report progress."""
    if progress_queue: progress_queue.put("Splitting audio into chunks...")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found for splitting: {audio_path}")

    os.makedirs(output_dir, exist_ok=True)
    chunks = []
    audio = None
    try:
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        start_time = 0
        chunk_number = 1

        while start_time < duration:
            end_time = min(start_time + chunk_duration, duration)
            if end_time <= start_time:
                break

            chunk_filename = f"audio_chunk_{chunk_number}.wav" # Simpler names
            chunk_path = os.path.join(output_dir, chunk_filename)
            # if progress_queue: progress_queue.put(f"Creating chunk {chunk_number}: {start_time:.1f}s - {end_time:.1f}s")

            # Write chunk
            sub_clip = audio.subclip(start_time, end_time)
            sub_clip.write_audiofile(chunk_path, codec='pcm_s16le', logger=None)
            #sub_clip.close() # Close subclip immediately

            chunks.append((chunk_path, start_time, end_time))
            start_time = end_time
            chunk_number += 1

        if progress_queue: progress_queue.put(f"Audio split into {len(chunks)} chunks.")
        if not chunks:
             if progress_queue: progress_queue.put("Warning: Audio splitting produced no chunks.")

    except Exception as e:
        print(f"Error during audio splitting ({audio_path}): {e}")
        for chunk_path, _, _ in chunks:
            safe_remove(chunk_path)
        if audio: audio.close()
        if progress_queue: progress_queue.put(f"❌ Error splitting audio: {e}")
        raise Exception(f"Error during audio splitting: {e}")
    finally:
        if audio:
            audio.close()

    return chunks

def transcribe_with_whisper(model, audio_path, progress_queue=None):
    """Transcribe audio using Whisper, report progress."""
    if progress_queue: progress_queue.put(f"Transcribing: {os.path.basename(audio_path)}...")
    if not os.path.exists(audio_path):
         raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")

    try:
        use_fp16 = model.device.type == 'cuda'
        result = model.transcribe(audio_path, fp16=use_fp16)
        # if progress_queue: progress_queue.put(f"Transcription complete for: {os.path.basename(audio_path)}")
        return result
    except Exception as e:
        if "CUDA out of memory" in str(e):
            err_msg = "❌ CUDA out of memory during transcription. Try a smaller model or shorter chunk duration."
            if progress_queue: progress_queue.put(err_msg)
            raise MemoryError(err_msg) from e
        else:
            err_msg = f"❌ Error during transcription for {os.path.basename(audio_path)}: {e}"
            if progress_queue: progress_queue.put(err_msg)
            raise Exception(err_msg) from e

def translate_subtitles(client, subtitles, target_language, custom_prompt_template, token_usage, progress_queue=None):
    """Translate SRT data using GPT, report progress, use custom prompt."""
    if not subtitles:
        return []

    texts_to_translate = [sub.get('text', '') for sub in subtitles]
    indexed_texts = [(i, text) for i, text in enumerate(texts_to_translate) if text and not text.isspace()]

    if not indexed_texts:
        if progress_queue: progress_queue.put("No text found in subtitles to translate.")
        return subtitles

    batch_prompt = "\n".join(f"{i+1}. {text}" for i, text in indexed_texts)
    num_to_translate = len(indexed_texts)

    if progress_queue: progress_queue.put(f"Translating {num_to_translate} subtitle segments...")

    system_prompt = custom_prompt_template.replace("{target_language}", target_language)
    if not ("Maintain the original numbering" in system_prompt or "Reply ONLY with the translated lines" in system_prompt):
         system_prompt += "\nMaintain the original numbering. Reply ONLY with the translated lines, each starting with its number and a period (e.g., '1. Translated text')."

    try:
        response = client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_prompt}
            ]
        )
        if response.usage:
            token_usage['prompt'] += response.usage.prompt_tokens
            token_usage['completion'] += response.usage.completion_tokens
        token_usage['calls'] += 1

        raw_translation = response.choices[0].message.content.strip()
        lines = raw_translation.split('\n')
        parsed_translations = {}
        for line in lines:
            if '. ' in line:
                try:
                    num_str, trans_text = line.split('. ', 1)
                    num = int(num_str)
                    parsed_translations[num] = trans_text.strip()
                except ValueError:
                    print(f"Warning: Could not parse line number from translation: {line}")
            elif line.strip():
                 print(f"Warning: Translation line format unexpected: {line}")

        translated_subtitles_list = subtitles[:]
        for original_index, _ in indexed_texts:
            translation = parsed_translations.get(original_index + 1)
            if translation:
                translated_subtitles_list[original_index]['text'] = translation
            else:
                 print(f"Warning: Missing translation for subtitle number {original_index + 1}. Keeping original.")

        if progress_queue: progress_queue.put("Translation finished.")
        return translated_subtitles_list

    except Exception as e:
        print(f"Batch translation error: {e}")
        if progress_queue: progress_queue.put(f"❌ Batch translation failed: {e}. Skipping translation for this chunk.")
        return subtitles

def read_srt(srt_path):
    """Read SRT file, robust parsing."""
    if not os.path.exists(srt_path):
        print(f"SRT file not found: {srt_path}")
        return []

    subtitles = []
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        entries = content.strip().replace('\r\n', '\n').split('\n\n')

        for entry_str in entries:
            lines = entry_str.strip().split('\n')
            if len(lines) >= 3:
                number = lines[0].strip()
                timing = lines[1].strip()
                text = "\n".join(lines[2:]).strip()
                if number.isdigit() and '-->' in timing and text:
                    text = text.replace('\n', ' ')
                    subtitles.append({
                        'number': number,
                        'timing': timing,
                        'text': text
                    })
                else:
                    print(f"Warning: Skipping malformed SRT entry in {os.path.basename(srt_path)}: {' | '.join(lines)}")
            elif entry_str.strip(): # Log blocks that are not empty but don't have 3+ lines
                 print(f"Warning: Skipping invalid SRT block in {os.path.basename(srt_path)}: '{entry_str}'")

    except Exception as e:
        print(f"Error reading SRT file {srt_path}: {e}")
        raise IOError(f"Could not read SRT file: {srt_path}") from e

    return subtitles

def write_srt(subtitles, output_path):
    """Write subtitles to SRT file, skipping empty text entries."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    entries_written = 0
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, subtitle in enumerate(subtitles):
                # Use provided number or generate sequential if missing
                number = subtitle.get('number', str(i + 1))
                timing = subtitle.get('timing')
                text = subtitle.get('text', '').strip()

                # Validate essential fields
                if not timing or '-->' not in timing:
                    print(f"Warning: Skipping subtitle entry {number} due to missing/invalid timing.")
                    continue
                if not text:
                    # print(f"Warning: Skipping subtitle entry {number} due to empty text.") # Can be noisy
                    continue

                f.write(f"{number}\n")
                f.write(f"{timing}\n")
                f.write(f"{text}\n\n")
                entries_written += 1
        print(f"SRT file written: {output_path} ({entries_written} entries)")
    except Exception as e:
        print(f"Error writing SRT file {output_path}: {e}")
        raise IOError(f"Could not write SRT file: {output_path}") from e

def process_audio_chunk(whisper_model, openai_client, chunk_path, chunk_number, total_chunks, start_time_offset, target_language, custom_prompt_template, token_usage, output_dir, progress_queue=None):
    """Process single chunk: transcribe, translate, generate SRTs, report progress, and return timings."""
    if progress_queue: progress_queue.put(f"Processing chunk {chunk_number}/{total_chunks}...")

    os.makedirs(output_dir, exist_ok=True)
    base_chunk_name = os.path.splitext(os.path.basename(chunk_path))[0]
    original_srt_path = os.path.join(output_dir, f'{base_chunk_name}_original.srt')
    translated_srt_path = os.path.join(output_dir, f'{base_chunk_name}_translated.srt')

    chunk_transcription_time = 0
    chunk_translation_time = 0

    try:
        # 1. Transcription
        t_start_transcribe = time.time()
        transcription = transcribe_with_whisper(whisper_model, chunk_path, progress_queue) # Pass queue down
        chunk_transcription_time = time.time() - t_start_transcribe
        
        if not transcription or 'segments' not in transcription or not transcription['segments']:
            if progress_queue: progress_queue.put(f"Warning: No segments found in transcription for chunk {chunk_number}.")
            write_srt([], original_srt_path)
            write_srt([], translated_srt_path)
            return original_srt_path, translated_srt_path, chunk_transcription_time, chunk_translation_time

        # 2. Create original SRT data
        original_subtitles_data = []
        for i, segment in enumerate(transcription['segments']):
            text = segment.get('text','').strip()
            if text:
                seg_start = segment['start']
                seg_end = segment['end']
                if seg_end <= seg_start: seg_end = seg_start + 0.1
                original_subtitles_data.append({
                    'number': str(i+1),
                    'timing': f"{format_time(seg_start)} --> {format_time(seg_end)}",
                    'text': text
                })
        write_srt(original_subtitles_data, original_srt_path)

        # 3. Translation
        original_subtitles_parsed = read_srt(original_srt_path)
        if not original_subtitles_parsed:
            if progress_queue: progress_queue.put(f"Warning: Original SRT for chunk {chunk_number} is empty. Skipping translation.")
            write_srt([], translated_srt_path)
            return original_srt_path, translated_srt_path, chunk_transcription_time, chunk_translation_time
        
        t_start_translate = time.time()
        translated_subtitles = translate_subtitles(
            openai_client,
            original_subtitles_parsed,
            target_language,
            custom_prompt_template,
            token_usage,
            progress_queue
        )
        chunk_translation_time = time.time() - t_start_translate
        write_srt(translated_subtitles, translated_srt_path)

        return original_srt_path, translated_srt_path, chunk_transcription_time, chunk_translation_time

    except Exception as e:
        print(f"Error processing chunk {chunk_number} ({os.path.basename(chunk_path)}): {e}")
        if progress_queue: progress_queue.put(f"❌ Error in chunk {chunk_number}: {e}")
        if not os.path.exists(original_srt_path): write_srt([], original_srt_path)
        if not os.path.exists(translated_srt_path): write_srt([], translated_srt_path)
        return original_srt_path, translated_srt_path, chunk_transcription_time, chunk_translation_time

def combine_srt_files(srt_paths, output_path, chunk_start_times, progress_queue=None):
    """Combine multiple SRT files, adjusting timestamps."""
    if progress_queue: progress_queue.put(f"Combining {len(srt_paths)} SRT files...")
    combined_subtitles = []
    master_subtitle_number = 1

    if len(srt_paths) != len(chunk_start_times):
        err_msg = f"❌ Error: Mismatch num SRT files ({len(srt_paths)}) and start times ({len(chunk_start_times)})."
        if progress_queue: progress_queue.put(err_msg)
        raise ValueError(err_msg)

    for i, (srt_path, chunk_start_offset) in enumerate(zip(srt_paths, chunk_start_times)):
        if not srt_path or not os.path.exists(srt_path) or os.path.getsize(srt_path) == 0:
            continue # Skip empty or non-existent files

        try:
            subtitles = read_srt(srt_path)
            for sub in subtitles:
                try:
                    if ' --> ' not in sub.get('timing', ''):
                        print(f"Warning: Invalid timing format in {os.path.basename(srt_path)}, subtitle {sub.get('number','?')}. Skipping.")
                        continue
                    start_str, end_str = sub['timing'].split(' --> ')

                    # Convert relative time to absolute time
                    start_seconds = srt_time_to_seconds(start_str) + chunk_start_offset
                    end_seconds = srt_time_to_seconds(end_str) + chunk_start_offset

                    # Simple validation/correction for timing
                    if end_seconds <= start_seconds:
                        end_seconds = start_seconds + 0.5 # Assign a minimum duration

                    new_timing = f"{format_time(start_seconds)} --> {format_time(end_seconds)}"
                    text = sub.get('text', '').strip()

                    if not text: continue # Skip empty text lines after adjustment

                    combined_subtitles.append({
                        'number': str(master_subtitle_number),
                        'timing': new_timing,
                        'text': text
                    })
                    master_subtitle_number += 1
                except Exception as e_inner:
                    # Log error for specific subtitle but continue combining others
                    print(f"Error processing subtitle entry {sub.get('number','?')} from {os.path.basename(srt_path)}: {e_inner}")
                    continue # Skip this specific subtitle entry

        except Exception as e_outer:
             # Log error reading/processing one SRT file but continue with others
             print(f"Error processing SRT file {os.path.basename(srt_path)} during combination: {e_outer}")
             continue # Skip this entire SRT file

    # Write the final combined file
    write_srt(combined_subtitles, output_path)
    if progress_queue: progress_queue.put(f"Combined SRT created: {os.path.basename(output_path)} ({master_subtitle_number - 1} entries)")


def embed_subtitles(video_path, srt_path, output_path, subtitle_style, progress_queue=None):
    """Embed subtitles into video, report progress."""
    if progress_queue: progress_queue.put("Embedding subtitles into video...")

    # --- Validation ---
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found for embedding: {video_path}")
    if not os.path.exists(srt_path) or os.path.getsize(srt_path) == 0:
        # Try to continue without embedding if SRT is missing/empty
        if progress_queue: progress_queue.put(f"Warning: SRT file '{os.path.basename(srt_path)}' not found or is empty. Skipping embedding.")
        return False # Indicate embedding was skipped

    # Check if SRT contains valid entries after parsing
    try:
        parsed_subs = read_srt(srt_path)
        if not parsed_subs:
             if progress_queue: progress_queue.put(f"Warning: SRT file '{os.path.basename(srt_path)}' contains no valid entries. Skipping embedding.")
             return False
    except Exception as e:
         if progress_queue: progress_queue.put(f"Error reading SRT before embedding: {e}. Skipping embedding.")
         return False
    # --- End Validation ---

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video = None
    final_clip = None
    temp_audio_path = os.path.join(os.path.dirname(output_path), f"temp_audio_{int(time.time()*1000)}.aac")
    
    # Create a UTF-8 compatible version of the SRT file
    utf8_srt_path = os.path.join(os.path.dirname(output_path), f"utf8_{os.path.basename(srt_path)}")
    
    try:
        # Create a UTF-8 compatible version of the SRT file
        with open(srt_path, 'r', encoding='utf-8') as original_srt:
            srt_content = original_srt.read()
        
        with open(utf8_srt_path, 'w', encoding='utf-8') as utf8_srt:
            utf8_srt.write(srt_content)
            
        # Parse subtitles directly instead of using SubtitlesClip
        subs = []
        with open(utf8_srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by blank lines to get subtitle blocks
        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # First line is number, second is time code
                time_code = lines[1]
                if '-->' not in time_code:
                    continue
                    
                start_time, end_time = time_code.split('-->')
                start_time = start_time.strip().replace(',', '.')
                end_time = end_time.strip().replace(',', '.')
                
                # Convert to seconds
                h, m, s = start_time.split(':')
                start_seconds = float(h) * 3600 + float(m) * 60 + float(s)
                
                h, m, s = end_time.split(':')
                end_seconds = float(h) * 3600 + float(m) * 60 + float(s)
                
                # Join remaining lines as text
                text = ' '.join(lines[2:])
                
                subs.append((start_seconds, end_seconds, text))
                
        video = VideoFileClip(video_path)
        video_width = video.w
        video_duration = video.duration
        
        # Create subtitle clips manually
        subtitle_clips = []
        for start, end, text in subs:
            duration = end - start
            if duration <= 0:
                continue
                
            txt_clip = TextClip(
                text,
                font=subtitle_style.get('font', 'Arial'),
                fontsize=subtitle_style.get('fontsize', 24),
                color=subtitle_style.get('color', 'white'),
                bg_color=subtitle_style.get('bg_color', 'black'),
                method='caption',
                size=(video_width * 0.9, None),
                align='center'
            )
            
            # Set position and duration
            txt_clip = txt_clip.set_position(subtitle_style.get('position', ('center', 'bottom')))
            txt_clip = txt_clip.set_start(start).set_duration(duration)
            subtitle_clips.append(txt_clip)
        
        # Combine video with subtitle clips
        final_clip = CompositeVideoClip([video] + subtitle_clips)
        
        # Write video file with specific parameters
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=temp_audio_path,
            remove_temp=True,
            preset='medium',
            ffmpeg_params=['-crf', '23'],
            threads=os.cpu_count() or 4,
            logger=None
        )
        
        if progress_queue: progress_queue.put(f"Video with embedded subtitles saved: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        err_msg = f"❌ Error embedding subtitles: {e}"
        print(err_msg)
        if progress_queue: progress_queue.put(err_msg)
        return False
    finally:
        if video: video.close()
        if final_clip: final_clip.close()
        safe_remove(temp_audio_path)
        safe_remove(utf8_srt_path)


# --- Main Processing Function (runs in thread) ---

def process_video_threaded(params, progress_queue, result_queue):
    """
    Main processing pipeline running in a separate thread.
    Uses helper functions and reports progress and timings via queue.
    """
    process_start_time = time.time()
    timings = {
        "openai_client_initialization": 0,
        "whisper_model_loading": 0,
        "audio_extraction": 0,
        "audio_splitting": 0,
        "total_transcription_time_chunks": 0,
        "total_translation_time_chunks": 0,
        "srt_combination_original": 0,
        "srt_combination_translated": 0,
        "subtitle_embedding": 0,
        "total_processing_time": 0,
        "thread_execution_time": 0
    }
    thread_start_time = time.time()

    token_usage = {'prompt': 0, 'completion': 0, 'calls': 0}
    combined_original_srt = None
    combined_translated_srt = None
    final_video_path = None
    full_audio_path = None
    whisper_model = None
    openai_client = None
    temp_audio_chunks_dir = os.path.join(TEMP_DIR, "audio_chunks")
    output_chunk_srts_dir = os.path.join(OUTPUT_DIR, "chunk_srts")


    try:
        video_path = params["video_path"]
        target_language = params["target_language"]
        whisper_model_size = params["whisper_model_size"]
        api_key = params["api_key"]
        custom_translation_prompt = params["custom_translation_prompt"]
        output_options = params["output_options"]

        progress_queue.put("🚀 Starting processing...")
        progress_queue.put(f"Video: {os.path.basename(video_path)}")
        progress_queue.put(f"Model: {whisper_model_size}, Language: {target_language}")
        progress_queue.put(f"Output: {output_options}")

        # --- Setup ---
        os.makedirs(temp_audio_chunks_dir, exist_ok=True)
        os.makedirs(output_chunk_srts_dir, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        t_start = time.time()
        progress_queue.put("Initializing OpenAI client...")
        try:
            openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}") from e
        timings["openai_client_initialization"] = time.time() - t_start
        
        t_start = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        progress_queue.put(f"Loading Whisper model '{whisper_model_size}' onto {device}...")
        try:
            whisper_model = whisper.load_model(whisper_model_size, device=device)
            progress_queue.put("Whisper model loaded.")
        except Exception as e:
            if "CUDA out of memory" in str(e):
                raise MemoryError(f"CUDA out of memory loading Whisper model. Try a smaller model.") from e
            raise Exception(f"Failed to load Whisper model: {e}") from e
        timings["whisper_model_loading"] = time.time() - t_start

        t_start = time.time()
        full_audio_path = extract_full_audio(video_path, TEMP_DIR, progress_queue)
        timings["audio_extraction"] = time.time() - t_start

        t_start = time.time()
        chunks_info = split_audio_into_chunks(full_audio_path, CHUNK_DURATION, temp_audio_chunks_dir, progress_queue)
        if not chunks_info:
            raise Exception("Audio splitting produced no chunks. Cannot proceed.")
        timings["audio_splitting"] = time.time() - t_start
        
        chunk_start_times = [info[1] for info in chunks_info]
        total_chunks = len(chunks_info)

        original_srt_paths = []
        translated_srt_paths = []
        progress_queue.put(f"Starting chunk processing (total {total_chunks})...")
        for i, (chunk_path, start_time_offset_for_chunk, _) in enumerate(chunks_info):
            orig_srt, trans_srt, chunk_transcribe_time, chunk_translate_time = process_audio_chunk(
                whisper_model, openai_client, chunk_path, i+1, total_chunks, start_time_offset_for_chunk,
                target_language, custom_translation_prompt, token_usage,
                output_chunk_srts_dir, progress_queue
            )
            original_srt_paths.append(orig_srt)
            translated_srt_paths.append(trans_srt)
            timings["total_transcription_time_chunks"] += chunk_transcribe_time
            timings["total_translation_time_chunks"] += chunk_translate_time

        progress_queue.put("All chunks processed.")

        base_output_name = os.path.splitext(os.path.basename(video_path))[0]
        
        t_start = time.time()
        combined_original_srt = os.path.join(OUTPUT_DIR, f"{base_output_name}_original_subtitles.srt")
        combine_srt_files(original_srt_paths, combined_original_srt, chunk_start_times, progress_queue)
        timings["srt_combination_original"] = time.time() - t_start

        t_start = time.time()
        combined_translated_srt = os.path.join(OUTPUT_DIR, f"{base_output_name}_{target_language}_subtitles.srt")
        combine_srt_files(translated_srt_paths, combined_translated_srt, chunk_start_times, progress_queue)
        timings["srt_combination_translated"] = time.time() - t_start
        
        embedding_successful = False
        if output_options in ['Embed', 'Both']:
            progress_queue.put("Preparing for subtitle embedding...")
            final_video_filename = f"{base_output_name}_with_{target_language}_subs.mp4"
            final_video_path = os.path.join(OUTPUT_DIR, final_video_filename)
            subtitle_style = {
                'font': SUBTITLE_FONT, 'fontsize': SUBTITLE_FONTSIZE, 'color': SUBTITLE_COLOR,
                'bg_color': SUBTITLE_BG_COLOR, 'position': SUBTITLE_POSITION
            }
            t_start = time.time()
            embedding_successful = embed_subtitles(
                video_path, combined_translated_srt, final_video_path,
                subtitle_style, progress_queue
            )
            timings["subtitle_embedding"] = time.time() - t_start
            if not embedding_successful:
                final_video_path = None
        else:
            progress_queue.put("Skipping subtitle embedding as per options.")
            timings["subtitle_embedding"] = 0 # Ei tehtud

        timings["total_processing_time"] = time.time() - process_start_time
        
        results = {
            'original_srt_path': combined_original_srt if os.path.exists(combined_original_srt) else None,
            'translated_srt_path': combined_translated_srt if os.path.exists(combined_translated_srt) else None,
            'final_video_path': final_video_path if final_video_path and os.path.exists(final_video_path) else None,
            'output_options': output_options,
            'token_usage': token_usage,
            'timings': timing
        }
        result_queue.put(results)

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"--- FATAL ERROR in processing thread ---\nError: {e}\n{tb_str}\n--- End Traceback ---")
        result_queue.put(f"❌ Error: {e}")
    finally:
        progress_queue.put("Cleaning up temporary files...")
        safe_remove(full_audio_path)
        safe_remove(temp_audio_chunks_dir)
        # safe_remove(output_chunk_srts_dir)
        progress_queue.put("🧹 Cleanup attempted.")
        
        timings["thread_execution_time"] = time.time() - thread_start_time

        
        end_time_total_thread_actual = time.time()
        progress_queue.put(f"Processing thread finished in {end_time_total_thread_actual - thread_start_time:.2f} seconds.")


# --- Streamlit UI---

st.set_page_config(layout="wide", page_title="Lecture Translator") # Muudetud pealkiri
st.title("🎓 Lecture Translator") # Muudetud pealkiri, lisatud ikoon
st.markdown("Upload a lecture video, get transcribed & translated subtitles (powered by Whisper & OpenAI).") # Üldistatud tekst

# --- State Management ---
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'progress_message' not in st.session_state:
    st.session_state.progress_message = ""
if 'results' not in st.session_state:
    st.session_state.results = None
if 'error' not in st.session_state:
    st.session_state.error = None
if 'current_video_file' not in st.session_state:
    st.session_state.current_video_file = None
if 'worker_thread' not in st.session_state:
    st.session_state.worker_thread = None
if 'token_summary' not in st.session_state:
    st.session_state.token_summary = ""
if 'timings_summary' not in st.session_state: # Uus olek ajakulu jaoks
    st.session_state.timings_summary = []


# --- Cleanup old directories on startup ---
if not st.session_state.processing: 
    safe_remove(TEMP_DIR)
    # safe_remove(OUTPUT_DIR) # Võib-olla soovid eelmise käivitamise tulemusi alles hoida
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Input Area ---
col1, col2 = st.columns([0.6, 0.4]) 

with col1:
    st.subheader("1. Upload Video")
    uploaded_file = st.file_uploader(
        "Select MP4 video file", # Muudetud tekst
        type=["mp4"],
        accept_multiple_files=False,
        disabled=st.session_state.processing,
        key="file_uploader"
    )
    if uploaded_file:
        if not st.session_state.processing:
            st.session_state.current_video_file = {
                "name": uploaded_file.name,
                "data": uploaded_file.getvalue() 
            }
            st.info(f"Selected: {uploaded_file.name}")
    elif st.session_state.current_video_file and not st.session_state.processing:
        st.info(f"Using previous file: {st.session_state.current_video_file['name']}")
    elif st.session_state.processing and st.session_state.current_video_file:
        st.info(f"Processing: {st.session_state.current_video_file['name']}")


    st.subheader("2. Settings")
    whisper_model = st.selectbox(
        "Transcription Model (Whisper)",
        AVAILABLE_WHISPER_MODELS,
        index=AVAILABLE_WHISPER_MODELS.index(DEFAULT_MODEL),
        disabled=st.session_state.processing,
        help="Larger models are more accurate but slower and require more memory/VRAM."
    )
    target_language = st.text_input(
        "Target Language for Translation",
        DEFAULT_LANGUAGE,
        disabled=st.session_state.processing,
        help="E.g., Spanish, French, German, Estonian"
    )
    output_options_ui = st.radio(
        "Output Options",
        ('Generate Translated SRT file only', 'Generate Translated SRT and Video with embedded subtitles'),
        index=1, # Vaikimisi "Both"
        disabled=st.session_state.processing,
        key="output_options_radio"
    )

with col2:
    st.subheader("3. Translation Tuning (Optional)")
    api_key = st.text_input(
        "OpenAI API Key (Required for Translation)",
        type="password",
        placeholder="sk-...",
        help="Your key is sent securely and used only for translation requests.",
        disabled=st.session_state.processing
    )
    custom_prompt_template = st.text_area(
        "Custom Translation Instructions",
        DEFAULT_TRANSLATION_PROMPT, 
        height=150,
        help="Advanced: Modify the instructions for the AI translator. Use '{target_language}' placeholder if needed.",
        disabled=st.session_state.processing
    )


# --- Action Button ---
st.divider()
start_button_placeholder = st.empty()

if not st.session_state.processing:
    start_button = start_button_placeholder.button(
        "🚀 Start Processing",
        type="primary",
        disabled=not st.session_state.current_video_file or not api_key,
        use_container_width=True
    )
else:
    start_button_placeholder.button("Processing...", disabled=True, use_container_width=True)
    start_button = False 


# --- Progress & Results Area ---
progress_bar_placeholder = st.empty()
status_text_placeholder = st.empty()

if st.session_state.processing:
    progress_bar_placeholder.progress(st.session_state.get('progress_value', 0.0))
    status_text_placeholder.info(st.session_state.get('progress_message', "Initializing..."))


if start_button:
    st.session_state.processing = True
    st.session_state.results = None
    st.session_state.error = None
    st.session_state.progress_message = "Initializing..."
    st.session_state.progress_value = 0.0
    st.session_state.token_summary = ""
    st.session_state.timings_summary = [] # Nulli ajakulu

    status_text_placeholder.info(st.session_state.progress_message)
    progress_bar_placeholder.progress(st.session_state.progress_value)

    video_data = st.session_state.current_video_file['data']
    video_filename = st.session_state.current_video_file['name']
    video_path = os.path.join(TEMP_DIR, video_filename)
    with open(video_path, "wb") as f:
        f.write(video_data)

    # Parem output_options loogika vastavalt UI valikutele
    if output_options_ui == 'Generate Translated SRT file only':
        output_choice_for_thread = 'SRT'
    elif output_options_ui == 'Generate Translated SRT and Video with embedded subtitles':
        output_choice_for_thread = 'Both'
    else: # Varuvariant, kui midagi peaks valesti minema
        output_choice_for_thread = 'SRT'


    processing_params = {
        "video_path": video_path,
        "target_language": target_language,
        "whisper_model_size": whisper_model,
        "api_key": api_key,
        "custom_translation_prompt": custom_prompt_template, 
        "output_options": output_choice_for_thread, # Kasuta korrigeeritud valikut
    }

    progress_queue = Queue()
    result_queue = Queue()

    st.session_state.worker_thread = Thread(target=process_video_threaded, args=(processing_params, progress_queue, result_queue), daemon=True)
    st.session_state.worker_thread.start()

    while st.session_state.worker_thread.is_alive():
        progress_value = st.session_state.get('progress_value', 0.0)
        while not progress_queue.empty():
            message = progress_queue.get()
            st.session_state.progress_message = message
            status_text_placeholder.info(st.session_state.progress_message)

            # Progressiriba uuendamise loogika (sama mis enne)
            if "Extracting audio" in message: progress_value = max(progress_value, 0.05)
            elif "Splitting audio" in message: progress_value = max(progress_value, 0.15)
            elif "chunk" in message and "/" in message:
                try:
                    parts = message.split("chunk ")[1].split("...")[0].split("/") # Robustsem parsemin
                    if len(parts) == 2:
                        progress_value = max(progress_value, 0.20 + (int(parts[0]) / int(parts[1])) * 0.50) 
                except: pass # Ignoreeri parse vigu, kui sõnumi formaat muutub
            elif "Combining SRT" in message: progress_value = max(progress_value, 0.75)
            elif "Embedding" in message: progress_value = max(progress_value, 0.85)
            elif "Cleaning up" in message: progress_value = max(progress_value, 0.98)
            elif "finished" in message: progress_value = 1.0
            
            st.session_state.progress_value = progress_value
            progress_bar_placeholder.progress(min(progress_value, 1.0))
        time.sleep(0.3)

    while not progress_queue.empty(): 
        message = progress_queue.get()
        st.session_state.progress_message = message
        if "finished" in message: 
            st.session_state.progress_value = 1.0
    progress_bar_placeholder.progress(st.session_state.get('progress_value', 1.0))
    status_text_placeholder.info(st.session_state.get('progress_message', 'Processing complete.'))

    result = result_queue.get() 

    if isinstance(result, str) and result.startswith("❌ Error:"):
        st.session_state.error = result
    else:
        st.session_state.results = result
        if result.get('token_usage'):
            tu = result['token_usage']
            total = tu['prompt'] + tu['completion']
            st.session_state.token_summary = f"Translation API: {tu['calls']} calls, {total} tokens ({tu['prompt']} prompt + {tu['completion']} completion)."
        
        # Lisa ajakulu kokkuvõte
        if result.get('timings'):
            timings_data = result['timings']
            st.session_state.timings_summary = [] # Tühjenda eelmine
            for key, value in timings_data.items():
                # Vorminda võtmenimed sõbralikumaks
                friendly_key = key.replace("_", " ").capitalize()
                st.session_state.timings_summary.append(f"{friendly_key}: {value:.2f} seconds")

    st.session_state.processing = False
    st.session_state.worker_thread = None
    st.rerun()


# --- Display Results ---
st.divider()
if st.session_state.results and not st.session_state.processing:
    st.subheader("✅ Results")
    results_data = st.session_state.results
    res_col1, res_col2 = st.columns(2)

    orig_srt = results_data.get('original_srt_path')
    trans_srt = results_data.get('translated_srt_path')
    final_video = results_data.get('final_video_path')
    options = results_data.get('output_options')

    with res_col1:
        st.write("**Download Files:**")
        if orig_srt: # Kuva alati originaal SRT allalaadimisnupp, kui see on olemas
            with open(orig_srt, "rb") as fp:
                st.download_button(
                    label="Download Original SRT (.srt)", data=fp,
                    file_name=os.path.basename(orig_srt), mime="application/x-subrip"
                )
        
        # Tõlgitud SRT allalaadimine sõltuvalt valikust (SRT või Both)
        if trans_srt and options in ['SRT', 'Both']:
            with open(trans_srt, "rb") as fp:
                st.download_button(
                    label=f"Download Translated SRT (.srt)", data=fp,
                    file_name=os.path.basename(trans_srt), mime="application/x-subrip"
                )
        
        # Video allalaadimine sõltuvalt valikust (Embed või Both)
        # Praeguse UI loogikaga on see alati 'Both', kui video genereeritakse
        if final_video and options in ['Embed', 'Both']: 
            with open(final_video, "rb") as fp:
                st.download_button(
                    label=f"Download Video with Subtitles (.mp4)", data=fp,
                    file_name=os.path.basename(final_video), mime="video/mp4"
                )
        
        if st.session_state.token_summary:
            st.caption(st.session_state.token_summary)
        
        # Kuva ajakulu kokkuvõte
        if st.session_state.timings_summary:
            st.write("**Processing Times:**")
            for item in st.session_state.timings_summary:
                st.caption(item)

    with res_col2:
        if final_video and options in ['Embed', 'Both']:
            st.write("**Video Preview:**")
            st.video(final_video)

elif st.session_state.error and not st.session_state.processing:
    st.error(f"{st.session_state.error}")

st.markdown("---")
st.caption("Ensure FFMPEG is installed and accessible in your system PATH for video processing. Subtitle embedding requires ImageMagick for some fonts/styles (install if text doesn't appear).")