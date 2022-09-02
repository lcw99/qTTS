import os

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

# we use the same path as this script as our training folder.
output_path = os.path.dirname(os.path.abspath(__file__))

# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
# You can also use a simple Dict to define the dataset and pass it to your custom formatter.
dataset_config = BaseDatasetConfig(
    name="0041_JBG",
    meta_file_train="0041_JBG.txt",
    path="/home/chang/bighard/AI/tts/dataset/multi_speaker/",
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=True,
)
# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.
config = GlowTTSConfig(
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    precompute_num_workers=8,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="korean_phoneme_cleaners_normalize",
    use_phonemes=True,
    phoneme_language="ko",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache_norm_ko_0041_JBG"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
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
    speaker_name = "0041_JBG"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        cnt = 0
        for line in ttf:
            cols = line.split("|")
            speaker = cols[1]
            wav_file = os.path.join(root_path, speaker, cols[0])
            speaker = cols[1]
            text = cols[2]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name})
            cnt += 1
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
