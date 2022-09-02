import os
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager

colab = False
if 'COLAB_GPU' in os.environ:
    colab = True

# we use the same path as this script as our training folder.
output_path = os.path.dirname(os.path.abspath(__file__))
num_worker=4
# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
# You can also use a simple Dict to define the dataset and pass it to your custom formatter.
data_path = "/home/chang/bighard/AI/tts/dataset/kss22050/"
if Path("/mnt/ramdisk/kss").is_dir():
    print("ramdisk exists...")
    data_path = "/mnt/ramdisk/kss22050"
phoneme_path = "/home/chang/bighard/AI/tts/dataset/kss/phoneme_cache_g2p_ko/"
batch_size = 16
if colab:
    data_path = "/content/drive/MyDrive/tts/dataset/kss/"
    phoneme_path = "/content/drive/MyDrive/tts/dataset/kss/phoneme_cache_g2p_ko"
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
    resample=colab,
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

config = FastSpeechConfig(
    run_name="fast_speech_kss_ko_phoneme_g2p",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    compute_f0=False,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="korean_phoneme_cleaners_g2p",
    use_phonemes=True,
    phoneme_language="ko",
    phoneme_cache_path=phoneme_path,
    precompute_num_workers=4,
    print_step=50,
    save_step=5000,
    print_eval=False,
    mixed_precision=True,
    max_seq_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
    test_sentences = [
        # "목소리를 만드는데는 오랜 시간이 걸린다, 인내심이 필요하다.",
        # "목소리가 되어라, 메아리가 되지말고.",
        # "철수야 미안하다. 아무래도 그건 못하겠다.",
        # "이 케익은 정말 맛있다. 촉촉하고 달콤하다.",
        # "1963년 11월 23일 이전",
    ],
)

config.model_args.use_pitch = False
config.model_args.use_aligner = True
# compute alignments
if not config.model_args.use_aligner:
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
    # TODO: make compute_attention python callable
    os.system(
        f"python TTS/bin/compute_attention_masks.py --model_path {model_path} --config_path {config_path} --dataset ljspeech --dataset_metafile metadata.csv --data_path ./recipes/ljspeech/LJSpeech-1.1/  --use_cuda true"
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

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, 
    eval_split=True, 
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter
)

# init the model
model = ForwardTTS(config, ap, tokenizer)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
