import os
from pathlib import Path
from re import T

from trainer import Trainer, TrainerArgs

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.speedy_speech_config import SpeedySpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

colab = False
if 'COLAB_GPU' in os.environ:
    colab = True

# we use the same path as this script as our training folder.
output_path = os.path.dirname(os.path.abspath(__file__))
num_worker=8
# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
# You can also use a simple Dict to define the dataset and pass it to your custom formatter.
data_path = "/home/chang/bighard/AI/tts/dataset/kss/"
if Path("/mnt/ramdisk/kss").is_dir():
    print("ramdisk exists...")
    data_path = "/mnt/ramdisk/kss"
phoneme_path = "/home/chang/bighard/AI/tts/dataset/kss/phoneme_cache_g2p_ko/"
batch_size = 32
if colab:
    data_path = "/content/drive/MyDrive/tts/dataset/kss/"
    phoneme_path = "/content/drive/MyDrive/tts/phoneme_cache_g2p_ko"
    batch_size = 32
    num_worker = 4
    
dataset_config = BaseDatasetConfig(
    name="kss_ko",
    meta_file_train="transcript.v.1.4.txt",
    language="ko-kr",
    path=data_path,
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=True,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = SpeedySpeechConfig(
    run_name="speedy_speech_ko",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=num_worker,
    num_eval_loader_workers=4,
    precompute_num_workers=num_worker,
    compute_input_seq_cache=True,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="korean_phoneme_cleaners_g2p",
    use_phonemes=True,
    phoneme_language="ko",
    phoneme_cache_path=phoneme_path,
    print_step=50,
    print_eval=False,
    mixed_precision=True,
    max_seq_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
# train_samples, eval_samples = load_tts_samples(
#     dataset_config,
#     eval_split=True,
#     eval_split_max_size=config.eval_split_max_size,
#     eval_split_size=config.eval_split_size,
# )
def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "KBSVoice"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        cnt = 0
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[1]
            if len(text) <= 5:
                continue            
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name})
            #cnt += 1
            #if cnt >= 10000:
            #if cnt >= 1000:
            #    break
    return items

train_samples, eval_samples = load_tts_samples(
    dataset_config, 
    eval_split=True, 
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter)

# init model
model = ForwardTTS(config, ap, tokenizer)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()
