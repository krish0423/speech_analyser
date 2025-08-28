# speech_analyser

# 🎤 Pronunciation Assessment Tool

A simple desktop app that analyzes your English pronunciation using AI and gives you feedback to improve your speech.

## What it does

- Records your voice or accepts audio files
- Analyzes pronunciation accuracy
- Shows which words need improvement with color coding
- Gives tips on speech rate and intonation
- Opens detailed reports in your web browser

## Requirements

- Python 3.11+
- Microphone (for recording)
- Internet connection (first time only, to download AI models)

## Installation

1. **Download this project**

   ```bash
   git clone https://github.com/yourusername/pronunciation-assessment-tool.git
   cd pronunciation-assessment-tool
   ```

2. **Create virtual environment**

   ```bash
   python -m venv pronunciation_env

   # Windows:
   pronunciation_env\Scripts\activate
   # Mac/Linux:
   source pronunciation_env/bin/activate
   ```

3. **Install packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Setup NLTK (run once)**

   ```bash
   python setup_nltk.py
   ```

5. **Run the app**
   ```bash
   python pronunciation_app.py
   ```

## How to use

1. Type a sentence to practice (or use the default)
2. Click "🎤 Start Recording" and speak the sentence
3. Click "🔍 Analyze" to get your results
4. View your detailed report in the web browser

## Files

- `pronunciation_app.py` - Main application
- `setup_nltk.py` - One-time setup for language data
- `requirements.txt` - List of required packages

## Troubleshooting

- **First run takes 3-5 minutes** - downloading AI models
- **"No module" errors** - make sure virtual environment is activated
- **Recording issues** - check microphone permissions
- **Need help?** - Create an issue on GitHub

## License

MIT License - feel free to use and modify!
