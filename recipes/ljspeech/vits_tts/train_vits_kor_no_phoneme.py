import os
import random
from re import T
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from dataclasses import dataclass, field

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
    data_path = "/mnt/ramdisk/kss"
phoneme_path = "/home/chang/bighard/AI/tts/dataset/kss/phoneme_cache_norm_ko/"
batch_size = 32
if colab:
    data_path = "/content/drive/MyDrive/tts/dataset/kss/"
    phoneme_path = "/content/drive/MyDrive/tts/dataset/kss/phoneme_cache_norm_ko"
    batch_size = 32
    num_worker = 4
    
dataset_config = BaseDatasetConfig(
    name="kss_ko",
    meta_file_train="transcript.v.1.4.txt",
    path=data_path,
)

audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None,
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_kss_ko_norm_no_phoneme",
    batch_size=batch_size,
    eval_batch_size=12,
    batch_group_size=5,
    num_loader_workers=num_worker,
    num_eval_loader_workers=2,
    precompute_num_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="korean_phoneme_cleaners_with_g2p_jamo_split",
    use_phonemes=False,
    phoneme_language="ko",
    #phoneme_cache_path=phoneme_path,
    phoneme_cache_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoneme_cache_norm_test"),
    compute_input_seq_cache=True,
    print_step=50,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    test_sentences = [
        ["???????????? ??????????????? ?????? ????????? ?????????, ???????????? ????????????."],
        ["???????????? ?????????, ???????????? ????????????."],
        ["????????? ????????????. ???????????? ?????? ????????????."],
        ["??? ????????? ?????? ?????????. ???????????? ????????????."],
        ["1963??? 11??? 23??? ??????"],
    ],
    #use_language_weighted_sampler=True,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz??????????????????????????????????????????????????????"+"???"+"????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????",
        punctuations="!??'(),-.:;??? ",
        phonemes=None,
    ),
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
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
        batch_coll = ttf.readlines()
        #batch_coll = random.choices(batch_coll, k=1000)
        for line in batch_coll:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[1]
            if len(text) <= 5:
                continue    
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name})
            cnt += 1
    return items

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config, 
    eval_split=True, 
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and ????
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
