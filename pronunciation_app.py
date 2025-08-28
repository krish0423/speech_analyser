import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import webbrowser
import tempfile
import os

# Core imports
import torch
import torchaudio
from g2p_en import G2p
import numpy as np
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
import seaborn as sns
import base64
from io import BytesIO
import matplotlib.patches as patches
import warnings
from itertools import groupby
import re
import requests
import sounddevice as sd

warnings.filterwarnings('ignore', category=UserWarning)

class SentencePronunciationAssessor:
    def __init__(self):
        print("🔧 Initializing Sentence Assessment System...")
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.alignment_model = self.bundle.get_model()
        self.labels = self.bundle.get_labels()
        self.cmudict = self._load_cmudict()
        if self.cmudict is None: 
            raise Exception("Could not load CMU Dictionary.")
        self.scoring_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.scoring_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.dictionary = {c: i for i, c in enumerate(self.labels)}
        self.unk_token_idx = self.dictionary.get('<unk>', self.dictionary['|'])
        self._load_references()
        print("✅ System initialized successfully!")

    def _load_cmudict(self):
        print("📚 Loading CMU Pronouncing Dictionary...")
        url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
        try:
            response = requests.get(url)
            text = response.text
            pronunciations = {}
            for line in text.splitlines():
                if not line.startswith(";;;"):
                    parts = line.split()
                    word = parts[0].lower()
                    phonemes = [re.sub(r'\d', '', p) for p in parts[1:]]
                    if "(" in word:
                        word = word.split("(")[0]
                    pronunciations[word] = phonemes
            return pronunciations
        except Exception as e:
            print(f"❌ Failed to load CMUdict: {e}")
            return None

    def _load_references(self):
        self.vowel_formants = {
            'AE': {'F1': 690, 'F2': 1650}, 'AH': {'F1': 650, 'F2': 1200}, 
            'AO': {'F1': 590, 'F2': 880}, 'ER': {'F1': 490, 'F2': 1350}, 
            'IH': {'F1': 400, 'F2': 1920}, 'IY': {'F1': 280, 'F2': 2250}, 
            'OW': {'F1': 460, 'F2': 1100}, 'UW': {'F1': 320, 'F2': 950}, 
            'AY': {'F1': 700, 'F2': 1500}, 'EY': {'F1': 400, 'F2': 1900}
        }
        self.vowels = self.vowel_formants.keys()

    def _get_aligned_words(self, waveform, sentence):
        transcript = sentence.upper().replace(" ", "|")
        with torch.inference_mode():
            emissions, _ = self.alignment_model(waveform.to("cpu"))
        
        tokens = [self.dictionary.get(c, self.unk_token_idx) for c in transcript]
        input_lengths = torch.tensor([emissions.shape[1]], dtype=torch.long)
        target_lengths = torch.tensor([len(tokens)], dtype=torch.long)
        
        aligned_tokens_tensor, _ = torchaudio.functional.forced_align(
            torch.log_softmax(emissions, dim=-1), 
            torch.tensor([tokens], dtype=torch.int32), 
            input_lengths, 
            target_lengths
        )
        
        aligned_tokens = aligned_tokens_tensor[0].tolist()
        ratio = waveform.shape[1] / emissions.shape[1] / self.bundle.sample_rate
        
        word_segments = []
        word_start_frame = 0
        separator_token_id = self.dictionary['|']
        break_indices = [i for i, token_id in enumerate(aligned_tokens) if token_id == separator_token_id]
        words = sentence.split()
        
        for i, word in enumerate(words):
            end_frame = break_indices[i] if i < len(break_indices) else len(aligned_tokens)
            word_segments.append({
                "word": word,
                "start_time": word_start_frame * ratio,
                "end_time": end_frame * ratio
            })
            word_start_frame = end_frame + 1
        
        return word_segments

    def _get_features_for_segment(self, segment, sr):
        features = {}
        if len(segment) < 20:
            return features
        
        try:
            sound = parselmouth.Sound(segment.astype(np.float64), sr)
            formant_obj = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f1 = call(formant_obj, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(formant_obj, "Get mean", 2, 0, 0, "Hertz")
            features['F1'] = f1 if not np.isnan(f1) else 0
            features['F2'] = f2 if not np.isnan(f2) else 0
        except:
            features['F1'], features['F2'] = 0, 0
        
        return features

    def _score_phoneme(self, phoneme, features):
        if phoneme in self.vowels:
            target = self.vowel_formants.get(phoneme)
            if not target or features.get('F1', 0) == 0:
                return 0.5
            f1_error = abs(features['F1'] - target['F1']) / target['F1']
            f2_error = abs(features['F2'] - target['F2']) / target['F2']
            return max(0.0, 1.0 - (f1_error + f2_error) / 2)
        return 0.8

    def _analyze_prosody(self, audio, sr, all_phonemes):
        duration = librosa.get_duration(y=audio, sr=sr)
        num_syllables = len([p for p in all_phonemes if p in self.vowels])
        syllables_per_second = num_syllables / duration if duration > 0 else 0
        rate_score = 1.0 - (abs(syllables_per_second - 4.5) / 4.5)
        
        sound = parselmouth.Sound(audio, sr)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values == 0] = np.nan
        pitch_std = np.nanstd(pitch_values) if not np.all(np.isnan(pitch_values)) else 0
        pitch_score = min(1.0, pitch_std / 30)
        
        return {
            'rate_score': max(0, rate_score),
            'pitch_score': max(0, pitch_score)
        }

    def assess(self, audio, sr, sentence):
        clean_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
        waveform = torch.from_numpy(audio).unsqueeze(0)
        word_segments = self._get_aligned_words(waveform, clean_sentence)
        
        all_phoneme_scores = []
        all_phonemes = []
        
        for segment in word_segments:
            word = segment['word']
            start_sample = int(segment['start_time'] * sr)
            end_sample = int(segment['end_time'] * sr)
            word_audio = audio[start_sample:end_sample]
            
            if word in self.cmudict:
                phonemes = self.cmudict[word]
                all_phonemes.extend(phonemes)
                word_duration = len(word_audio) / sr
                
                if not phonemes or word_duration < 0.01:
                    continue
                
                phoneme_duration = word_duration / len(phonemes)
                for i, p in enumerate(phonemes):
                    p_start = int(i * phoneme_duration * sr)
                    p_end = int((i + 1) * phoneme_duration * sr)
                    features = self._get_features_for_segment(word_audio[p_start:p_end], sr)
                    score = self._score_phoneme(p, features)
                    all_phoneme_scores.append({
                        'word': word, 
                        'phoneme': p, 
                        'score': score
                    })
        
        prosody_scores = self._analyze_prosody(audio, sr, all_phonemes)
        
        # ASR accuracy
        input_values = self.scoring_processor(audio, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            logits = self.scoring_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        recognized_text = self.scoring_processor.decode(predicted_ids[0]).lower()
        
        # Calculate edit distance
        s1, s2 = clean_sentence.replace(" ", ""), recognized_text.replace(" ", "")
        m, n = len(s1), len(s2)
        dp = [[i+j for j in range(n+1)] for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        accuracy_score = max(0, 1-(dp[m][n]/max(m,n,1)))
        overall_phoneme_score = np.mean([p['score'] for p in all_phoneme_scores]) if all_phoneme_scores else 0
        overall_score = np.mean([accuracy_score, prosody_scores['rate_score'], prosody_scores['pitch_score'], overall_phoneme_score])
        
        results = {
            'overall_score': overall_score,
            'accuracy_score': accuracy_score,
            'phoneme_score': overall_phoneme_score,
            'prosody': prosody_scores,
            'phoneme_scores': all_phoneme_scores,
            'recognized_text': recognized_text
        }
        
        return self.create_html_report(clean_sentence, results)

    def create_html_report(self, sentence, results):
        def get_color(score):
            if score > 0.8: return '#28a745'
            if score > 0.6: return '#ffc107'
            return '#dc3545'
        
        score_percent = results['overall_score'] * 100
        circumference = 2 * np.pi * 45
        offset = circumference - (score_percent / 100 * circumference)
        score_color = get_color(results['overall_score'])
        
        svg_chart = f"""<svg width="120" height="120" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#e6e6e6" stroke-width="10" />
            <circle cx="50" cy="50" r="45" fill="none" stroke="{score_color}" stroke-width="10" 
                    stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" 
                    stroke-linecap="round" transform="rotate(-90 50 50)" />
            <text x="50" y="55" text-anchor="middle" font-size="24" font-weight="bold" fill="{score_color}">
                {score_percent:.0f}%
            </text>
        </svg>"""
        
        # Word coloring
        word_scores = {}
        for p in results['phoneme_scores']:
            word = p['word']
            score = p['score']
            if word not in word_scores:
                word_scores[word] = []
            word_scores[word].append(score)
        
        colored_sentence_html = ""
        for word in sentence.split():
            avg_score = np.mean(word_scores.get(word, [0]))
            color = get_color(avg_score)
            colored_sentence_html += f'<span class="word" style="background-color:{color};">{word}</span> '
        
        prosody = results['prosody']
        prosody_html = f"""
        <div class="meter-container">
            <span>Clarity</span>
            <div class="meter-bar">
                <div style="width:{results['accuracy_score']*100}%; background-color:{get_color(results['accuracy_score'])}"></div>
            </div>
            <span>{results['accuracy_score']:.0%}</span>
        </div>
        <div class="meter-container">
            <span>Speech Rate</span>
            <div class="meter-bar">
                <div style="width:{prosody['rate_score']*100}%; background-color:{get_color(prosody['rate_score'])}"></div>
            </div>
            <span>{prosody['rate_score']:.0%}</span>
        </div>
        <div class="meter-container">
            <span>Intonation</span>
            <div class="meter-bar">
                <div style="width:{prosody['pitch_score']*100}%; background-color:{get_color(prosody['pitch_score'])}"></div>
            </div>
            <span>{prosody['pitch_score']:.0%}</span>
        </div>
        """
        
        # Recommendations
        recs = []
        if results['prosody']['rate_score'] < 0.7:
            recs.append("Your speech pace could be more natural. Try adjusting your speed.")
        if results['prosody']['pitch_score'] < 0.7:
            recs.append("Your voice sounds a bit monotone. Use more pitch variation to sound more expressive.")
        if results['accuracy_score'] < 0.8:
            recs.append(f"Focus on clearer articulation. The system heard: '{results['recognized_text']}'.")
        if not recs:
            recs.append("Excellent work! Your pronunciation is clear and natural.")
        
        recs_html = "<ul>" + "".join([f"<li>{rec}</li>" for rec in recs]) + "</ul>"
        
        report_html = f"""
        <style>
        .report-card {{
            font-family: sans-serif;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 24px;
            max-width: 700px;
            margin: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }}
        .report-header {{
            display: flex;
            align-items: center;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 16px;
            margin-bottom: 16px;
        }}
        .header-text {{
            margin-left: 20px;
        }}
        .header-text h2 {{
            margin: 0;
            color: #333;
        }}
        .header-text p {{
            margin: 0;
            color: #777;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: bold;
            color: #444;
            margin-top: 24px;
            margin-bottom: 12px;
        }}
        .colored-sentence {{
            font-size: 22px;
            font-weight: 500;
            line-height: 1.8;
        }}
        .word {{
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            margin-right: 4px;
        }}
        .meter-container {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 14px;
            color: #555;
        }}
        .meter-container span {{
            width: 100px;
        }}
        .meter-bar {{
            flex-grow: 1;
            height: 16px;
            background-color: #e6e6e6;
            border-radius: 8px;
            overflow: hidden;
        }}
        .meter-bar div {{
            height: 100%;
            border-radius: 8px;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
            color: #333;
        }}
        </style>
        <div class="report-card">
            <div class="report-header">
                {svg_chart}
                <div class="header-text">
                    <h2>Pronunciation Score</h2>
                    <p>Your overall performance for this sentence.</p>
                </div>
            </div>
            <div class="section-title">Word-by-Word Accuracy</div>
            <div class="colored-sentence">{colored_sentence_html}</div>
            <div class="section-title">Prosody Analysis (Rhythm & Intonation)</div>
            {prosody_html}
            <div class="section-title">💡 Recommendations</div>
            <div class="recommendations">{recs_html}</div>
        </div>
        """
        
        return report_html


class PronunciationApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pronunciation Assessment Tool")
        self.root.geometry("800x600")
        
        # Initialize assessor in background
        self.assessor = None
        self.audio_data = None
        self.sample_rate = 16000
        self.recording = False
        self.recorded_audio = []
        
        self.setup_ui()
        self.init_assessor()
    
    def setup_ui(self):
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=10)
        ttk.Label(title_frame, text="🎤 Pronunciation Assessment Tool", 
                 font=('Arial', 16, 'bold')).pack()
        
        # Sentence input
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=10, padx=20, fill='x')
        ttk.Label(input_frame, text="Enter sentence to practice:").pack(anchor='w')
        self.sentence_var = tk.StringVar(value="speech analysis is fun")
        self.sentence_entry = ttk.Entry(input_frame, textvariable=self.sentence_var, 
                                       font=('Arial', 12), width=60)
        self.sentence_entry.pack(fill='x', pady=5)
        
        # Audio controls
        audio_frame = ttk.Frame(self.root)
        audio_frame.pack(pady=10)
        
        self.record_btn = ttk.Button(audio_frame, text="🎤 Start Recording", 
                                    command=self.toggle_recording)
        self.record_btn.pack(side='left', padx=5)
        
        self.upload_btn = ttk.Button(audio_frame, text="📤 Upload Audio File", 
                                    command=self.upload_audio)
        self.upload_btn.pack(side='left', padx=5)
        
        self.analyze_btn = ttk.Button(audio_frame, text="🔍 Analyze", 
                                     command=self.analyze_audio, state='disabled')
        self.analyze_btn.pack(side='left', padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to record or upload audio")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
        # Results area
        results_frame = ttk.Frame(self.root)
        results_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill='both', expand=True)
        
        self.results_text = tk.Text(text_frame, wrap='word', height=20, width=80)
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def init_assessor(self):
        def load_assessor():
            try:
                self.status_var.set("Loading AI models... This may take a few minutes.")
                self.assessor = SentencePronunciationAssessor()
                self.status_var.set("✅ Ready to record or upload audio")
                self.record_btn.configure(state='normal')
                self.upload_btn.configure(state='normal')
            except Exception as e:
                self.status_var.set(f"❌ Error loading models: {str(e)}")
        
        # Load assessor in background thread
        thread = threading.Thread(target=load_assessor, daemon=True)
        thread.start()
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        if self.assessor is None:
            messagebox.showerror("Error", "AI models not loaded yet. Please wait.")
            return
        
        self.recording = True
        self.recorded_audio = []
        self.record_btn.configure(text="🔴 Stop Recording")
        self.status_var.set("🎙️ Recording... Speak now!")
        
        def record_callback(indata, frames, time, status):
            if self.recording:
                self.recorded_audio.append(indata.copy())
        
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, 
                                   callback=record_callback, dtype=np.float32)
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        
        if self.recorded_audio:
            self.audio_data = np.concatenate(self.recorded_audio, axis=0).flatten()
            self.status_var.set(f"✅ Recording complete! Duration: {len(self.audio_data)/self.sample_rate:.2f}s")
            self.analyze_btn.configure(state='normal')
        else:
            self.status_var.set("❌ No audio recorded")
        
        self.record_btn.configure(text="🎤 Start Recording")
    
    def upload_audio(self):
        if self.assessor is None:
            messagebox.showerror("Error", "AI models not loaded yet. Please wait.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg")]
        )
        
        if file_path:
            try:
                self.status_var.set("Loading audio file...")
                audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                self.audio_data = audio
                self.status_var.set(f"✅ Audio loaded! Duration: {len(audio)/sr:.2f}s")
                self.analyze_btn.configure(state='normal')
            except Exception as e:
                messagebox.showerror("Error", f"Could not load audio file: {str(e)}")
                self.status_var.set("❌ Error loading audio file")
    
    def analyze_audio(self):
        if self.audio_data is None or self.assessor is None:
            messagebox.showerror("Error", "No audio data or assessor not ready")
            return
        
        sentence = self.sentence_var.get().strip()
        if not sentence:
            messagebox.showerror("Error", "Please enter a sentence to analyze")
            return
        
        self.status_var.set("🔍 Analyzing pronunciation...")
        self.analyze_btn.configure(state='disabled')
        
        def analyze_thread():
            try:
                html_report = self.assessor.assess(self.audio_data, self.sample_rate, sentence)
                
                # Save HTML report and open in browser
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    f.write(html_report)
                    temp_path = f.name
                
                # Open in default browser
                webbrowser.open(f'file://{temp_path}')
                
                # Also display text summary
                self.root.after(0, lambda: self.show_text_summary(html_report))
                self.root.after(0, lambda: self.status_var.set("✅ Analysis complete! Report opened in browser."))
                self.root.after(0, lambda: self.analyze_btn.configure(state='normal'))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("❌ Analysis failed"))
                self.root.after(0, lambda: self.analyze_btn.configure(state='normal'))
        
        thread = threading.Thread(target=analyze_thread, daemon=True)
        thread.start()
    
    def show_text_summary(self, html_report):
        # Extract key information for text display
        self.results_text.delete(1.0, tk.END)
        
        # Parse the HTML to extract key metrics (simplified)
        if "overall_score" in str(self.assessor.__dict__ if self.assessor else {}):
            self.results_text.insert(tk.END, "📊 PRONUNCIATION ANALYSIS COMPLETE\n")
            self.results_text.insert(tk.END, "=" * 40 + "\n\n")
            self.results_text.insert(tk.END, "🌐 Detailed report opened in your web browser!\n\n")
            self.results_text.insert(tk.END, "📝 The report includes:\n")
            self.results_text.insert(tk.END, "• Overall pronunciation score with visual chart\n")
            self.results_text.insert(tk.END, "• Word-by-word accuracy highlighting\n")
            self.results_text.insert(tk.END, "• Prosody analysis (rhythm & intonation)\n")
            self.results_text.insert(tk.END, "• Personalized recommendations\n\n")
            self.results_text.insert(tk.END, "💡 Try recording the same sentence multiple times to track your improvement!")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("🚀 Starting Pronunciation Assessment Tool...")
    app = PronunciationApp()
    app.run()