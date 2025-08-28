import nltk
import os

# Create NLTK data directory
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK data
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
print("NLTK data downloaded successfully!")