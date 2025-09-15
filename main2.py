import os
import re
import glob
import dashscope


MODEL = 'qwen3-asr-flash'
BAI_LIAN_KEY = ''
DATA_DIR = 'data'
SAMPLE1_DIR = os.path.join(DATA_DIR, 'sample1')
SAMPLE1_PATH = os.path.join(DATA_DIR, 'sample1_qwen3_asr.txt')
SAMPLE2_DIR = os.path.join(DATA_DIR, 'sample2')
SAMPLE2_PATH = os.path.join(DATA_DIR, 'sample2_qwen3_asr.txt')
SAMPLE3_DIR = os.path.join(DATA_DIR, 'sample3')
SAMPLE3_PATH = os.path.join(DATA_DIR, 'sample3_qwen3_asr.txt')
SAMPLE4_DIR = os.path.join(DATA_DIR, 'sample4')
SAMPLE4_PATH = os.path.join(DATA_DIR, 'sample4_qwen3_asr.txt')
SAMPLE5_DIR = os.path.join(DATA_DIR, 'sample5')
SAMPLE5_PATH = os.path.join(DATA_DIR, 'sample5_qwen3_asr.txt')


def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else - 1


def run():
    wav_files = glob.glob(os.path.join(SAMPLE2_DIR, '*.wav'))
    wav_files = sorted(wav_files, key=lambda x: extract_number(os.path.basename(x)))
    with open(SAMPLE2_PATH, 'w', encoding='utf-8') as f:
        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            print(f'Processing: {filename}')
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'audio': wav_path},
                    ]
                }
            ]
            response = dashscope.MultiModalConversation.call(
                api_key=BAI_LIAN_KEY,
                model=MODEL,
                messages=messages,
                result_format='message',
                asr_options={
                    "language": "en",  # 可选，若已知音频的语种，可通过该参数指定待识别语种，以提升识别准确率
                    "enable_lid": True,
                    "enable_itn": False
                }
            )
            print(response)
            text = response['output']['choices'][0]['message']['content'][0]['text']
            print('Result:', text)
            print()
            f.write(f'{filename}\n{text}\n\n')


if __name__ == '__main__':
    run()
