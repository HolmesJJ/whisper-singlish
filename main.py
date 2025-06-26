import os
import glob

from transformers import pipeline
from transformers import AutoProcessor
from transformers import AutoModelForSpeechSeq2Seq


MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'whisper-small-singlish-122k')
DATA_DIR = 'data'


def run():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH)
    pipe = pipeline('automatic-speech-recognition', model=model, tokenizer=tokenizer,
                    feature_extractor=feature_extractor)
    wav_files = glob.glob(os.path.join(DATA_DIR, '*.wav'))
    for wav_path in wav_files:
        print(f'Processing: {os.path.basename(wav_path)}')
        result = pipe(wav_path)
        print('Result:', result['text'])
        print('\n')


if __name__ == '__main__':
    run()
