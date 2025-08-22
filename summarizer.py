
import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# --- Third-party libraries ---
# We're wrapping imports in a try-except block to provide a clean error
# message if a dependency is missing.
try:
    import torch
    from textblob import TextBlob
    from tqdm import tqdm
    from transformers import BartForConditionalGeneration, BartTokenizer
except ImportError:
    print("❌ A required library is missing.")
    print("   Please run: pip install torch transformers textblob tqdm")
    sys.exit(1)

# ==================================
# 1. CONFIGURATION
# ==================================

# Using a simple dataclass for essential settings. No need for multiple complex configs.
@dataclass
class Config:
    """Stores all essential configuration settings."""
    summarizer_model_name: str = "facebook/bart-large-cnn"
    max_input_words: int = 1024  # Max words per chunk for the summarizer
    max_file_size_mb: int = 5  # Safety limit for input files
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
# ==================================
# 2. CUSTOM EXCEPTIONS
# ==================================

# Custom exceptions make error handling much clearer than generic `except Exception`.
class ModelLoadError(Exception):
    """Raised when the ML model fails to load."""
    pass

class TextProcessingError(Exception):
    """Raised for issues during text processing (e.g., empty input)."""
    pass

# ==================================
# 3. CORE LOGIC
# ==================================

# Global variables for lazy loading. We don't want to load these huge models
# into memory unless we absolutely have to.
_SUMMARIZER_MODEL = None
_SUMMARIZER_TOKENIZER = None

def _lazy_load_summarizer(model_name: str, device: str):
    """
    Loads the summarization model and tokenizer only when first needed.
    This massively improves startup time and reduces initial memory usage.
    """
    global _SUMMARIZER_MODEL, _SUMMARIZER_TOKENIZER
    
    # If already loaded, do nothing.
    if _SUMMARIZER_MODEL and _SUMMARIZER_TOKENIZER:
        return

    try:
        print(f"INFO: Loading summarization model '{model_name}' for the first time...")
        
        # Use tqdm for a nice progress bar during download/loading
        with tqdm(total=2, desc="Initializing model") as pbar:
            _SUMMARIZER_TOKENIZER = BartTokenizer.from_pretrained(model_name)
            pbar.update(1)
            _SUMMARIZER_MODEL = BartForConditionalGeneration.from_pretrained(model_name).to(device)
            pbar.update(1)
        
        print(f"INFO: Model loaded successfully onto '{device}'.")

    except Exception as e:
        # Catching a broad exception here because many things can go wrong with HuggingFace downloads.
        # We then raise our own custom, clearer exception.
        raise ModelLoadError(f"Failed to load model '{model_name}'. Check your internet connection. Error: {e}")

def analyze_emotion(text: str) -> dict:
    """
    Get basic sentiment (positive/negative/neutral) from text using TextBlob.
    
    It's simple, fast, and doesn't need a huge transformer model.
    
    Args:
        text: Input text to analyze.
        
    Returns:
        A dictionary with sentiment info.
        Example: {"sentiment": "positive", "polarity": 0.85}
    """
    # FIXME: This is a very basic sentiment analysis. It can't detect complex
    # things like sarcasm or nuanced emotional states. But it's good enough for our purpose.
    if not text:
        return {"sentiment": "neutral", "polarity": 0.0}

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {"sentiment": sentiment, "polarity": polarity}
    except Exception:
        # If TextBlob fails for some reason, we don't want the whole script to crash.
        # Just log it and default to neutral.
        logging.warning("Emotion analysis failed. Defaulting to neutral.")
        return {"sentiment": "neutral", "polarity": 0.0}

def generate_summary(text: str, emotion: dict, max_length: int) -> str:
    """
    Generates a summary using the BART model, guided by the detected emotion.

    Args:
        text: The original text.
        emotion: The emotion analysis result from `analyze_emotion`.
        max_length: The target maximum length of the summary.

    Returns:
        The generated summary string.
    """
    cfg = Config()
    _lazy_load_summarizer(cfg.summarizer_model_name, cfg.device)

    # This is a bit of a hack, but adding a prefix can sometimes guide the model
    # to generate a summary with the right tone. It's a pragmatic compromise.
    emotion_prefix = {
        "positive": "An uplifting summary of the key points: ",
        "negative": "A critical summary of the concerning points: ",
        "neutral": "An objective summary of the main points: "
    }.get(emotion.get("sentiment", "neutral"), "")
    
    # Prepend the prefix to the original text
    text_with_prefix = emotion_prefix + text
    
    # Simple chunking for long texts. We just split by words.
    # TODO: A more clever chunking strategy would be to split by sentences or paragraphs.
    words = text_with_prefix.split()
    chunks = [' '.join(words[i:i + cfg.max_input_words]) for i in range(0, len(words), cfg.max_input_words)]
    
    summary_parts = []
    print(f"INFO: Processing text in {len(chunks)} chunk(s)...")
    try:
        for chunk in tqdm(chunks, desc="Summarizing"):
            inputs = _SUMMARIZER_TOKENIZER.encode(chunk, return_tensors="pt", max_length=cfg.max_input_words, truncation=True).to(cfg.device)
            
            # These parameters are a good starting point for decent summaries.
            summary_ids = _SUMMARIZER_MODEL.generate(
                inputs,
                max_length=max_length // len(chunks) + 20, # Rough allocation per chunk
                min_length=15,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            part = _SUMMARIZER_TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
            summary_parts.append(part)

        # If we had multiple chunks, we summarize the summaries to make it coherent.
        # This is a common and effective technique for long documents.
        if len(summary_parts) > 1:
            combined_summary = " ".join(summary_parts)
            final_inputs = _SUMMARIZER_TOKENIZER.encode(combined_summary, return_tensors="pt", max_length=cfg.max_input_words, truncation=True).to(cfg.device)
            final_summary_ids = _SUMMARIZER_MODEL.generate(
                final_inputs, max_length=max_length, min_length=30, num_beams=4, early_stopping=True
            )
            return _SUMMARIZER_TOKENIZER.decode(final_summary_ids[0], skip_special_tokens=True)
        else:
            return summary_parts[0] if summary_parts else ""

    except Exception as e:
        raise TextProcessingError(f"Failed during summary generation. Error: {e}")
    finally:
        # Crucial step: clean up GPU memory after we're done to prevent memory leaks.
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

def load_text_from_source(file_path: Optional[str], text_input: Optional[str]) -> str:
    """Loads text from either a file or a direct string input."""
    cfg = Config()
    if file_path:
        try:
            p = Path(file_path)
            file_size_mb = p.stat().st_size / (1024 * 1024)
            if file_size_mb > cfg.max_file_size_mb:
                raise TextProcessingError(f"File size ({file_size_mb:.2f}MB) exceeds the limit of {cfg.max_file_size_mb}MB.")
            return p.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise TextProcessingError(f"Input file not found: {file_path}")
        except Exception as e:
            # Catching other potential IO errors
            raise TextProcessingError(f"Could not read file: {e}")
    elif text_input:
        return text_input
    return "" # Should not happen with argparse, but good for safety

# ==================================
# 4. COMMAND-LINE INTERFACE
# ==================================

def main():
    """The main entry point for the script."""
    
    # Simple CLI setup using argparse. Only the essential options.
    parser = argparse.ArgumentParser(
        description="A practical, emotion-aware text summarizer.",
        epilog="Example: python your_script_name.py -i my_article.txt -l 150"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input", type=str, help="Path to the input text file.")
    input_group.add_argument("-t", "--text", type=str, help="Direct text input to summarize.")
    
    parser.add_argument("-o", "--output", type=str, help="Path to save the output summary. Prints to console if not provided.")
    parser.add_argument("-l", "--max-length", type=int, default=150, help="Maximum length of the summary in words. (Default: 150)")

    args = parser.parse_args()
    start_time = time.time()
    
    try:
        # --- Step 1: Load Text ---
        print("INFO: Loading text...")
        original_text = load_text_from_source(args.input, args.text)
        
        # Basic input validation
        if not original_text.strip():
            raise TextProcessingError("Input text is empty. Nothing to summarize.")
        
        original_word_count = len(original_text.split())
        if original_word_count < 30:
            print("INFO: Text is very short. Returning original text as summary.")
            summary = original_text
        else:
            # --- Step 2: Analyze Emotion ---
            print("INFO: Analyzing emotional tone...")
            emotion = analyze_emotion(original_text)
            print(f"INFO: Detected sentiment: {emotion['sentiment']} (Polarity: {emotion['polarity']:.2f})")
            
            # --- Step 3: Generate Summary ---
            summary = generate_summary(original_text, emotion, args.max_length)

        # --- Step 4: Display Results ---
        summary_word_count = len(summary.split())
        compression_ratio = (1 - summary_word_count / original_word_count) * 100 if original_word_count > 0 else 0
        
        # Human-readable output format
        output_content = (
            f"--- Summary Result ---\n"
            f"\n[Original Text]\n{original_text[:200]}...\n"
            f"\n[Generated Summary]\n{summary}\n"
            f"\n--- Analytics ---\n"
            f"Time Taken: {time.time() - start_time:.2f} seconds\n"
            f"Original Word Count: {original_word_count}\n"
            f"Summary Word Count: {summary_word_count}\n"
            f"Compression: {compression_ratio:.1f}%\n"
        )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"\n✅ Summary successfully saved to '{args.output}'")
            print(f"   Original words: {original_word_count}, Summary words: {summary_word_count}")
        else:
            print(output_content)

    # This is where we provide user-friendly error messages.
    except (ModelLoadError, TextProcessingError) as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        # For developers, it's useful to know where it happened.
        logging.exception("Traceback:")
        sys.exit(1)

if __name__ == "__main__":
    # We only run the main function if the script is executed directly.
    main()
