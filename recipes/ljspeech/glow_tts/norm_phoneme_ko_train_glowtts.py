import os
import random
from pathlib import Path

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

data_path = "/home/chang/tts_dataset/kss22050/"
phoneme_path = "/home/chang/tts_dataset/kss22050/phoneme_cache_norm_ko/"
num_worker=4
batch_size = 32
    
dataset_config = BaseDatasetConfig(
    name="kss_ko",
    meta_file_train="transcript.v.1.4.txt",
    language="ko-kr",
    path=data_path,
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    #resample=True,
)
# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.
config = GlowTTSConfig(
    run_name="glow_tts_ko_phoneme_norm",
    audio=audio_config,
    batch_size=batch_size,
    eval_batch_size=16,
    num_loader_workers=num_worker,
    num_eval_loader_workers=4,
    precompute_num_workers=num_worker,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="korean_phoneme_cleaners_normalize",
    use_phonemes=True,
    phoneme_language="ko",
    phoneme_cache_path=phoneme_path,
    print_step=50,
    save_step=5000,
    print_eval=False,
    mixed_precision=True,
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
config.encoder_params["num_heads"] = 4
config.encoder_params["num_layers"] = 12

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
        data = ttf.readlines()
        #data = random.choices(data, k=3000)
        for line in data:
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

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()
