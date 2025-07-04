import os
import re
import glob

from transformers import pipeline
from transformers import AutoProcessor
from transformers import AutoModelForSpeechSeq2Seq


MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'whisper-small')
DATA_DIR = 'data'
SAMPLE1_DIR = os.path.join(DATA_DIR, 'sample1')
SAMPLE1_PATH = os.path.join(DATA_DIR, 'sample1.txt')
SAMPLE2_DIR = os.path.join(DATA_DIR, 'sample2')
SAMPLE2_PATH = os.path.join(DATA_DIR, 'sample2.txt')
SAMPLE3_DIR = os.path.join(DATA_DIR, 'sample3')
SAMPLE3_PATH = os.path.join(DATA_DIR, 'sample3.txt')
SAMPLE4_DIR = os.path.join(DATA_DIR, 'sample4')
SAMPLE4_PATH = os.path.join(DATA_DIR, 'sample4.txt')


def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else - 1


def run():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH)
    pipe = pipeline('automatic-speech-recognition', model=model, tokenizer=tokenizer,
                    feature_extractor=feature_extractor)
    wav_files = glob.glob(os.path.join(SAMPLE1_DIR, '*.wav'))
    wav_files = sorted(wav_files, key=lambda x: extract_number(os.path.basename(x)))
    with open(SAMPLE1_PATH, 'w', encoding='utf-8') as f:
        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            print(f'Processing: {filename}')
            result = pipe(wav_path)
            text = result['text']
            print('Result:', text)
            print()
            f.write(f'{filename}\n{text}\n\n')


if __name__ == '__main__':
    run()
