import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

colab = False
if 'COLAB_GPU' in os.environ:
    colab = True

output_path = os.path.dirname(os.path.abspath(__file__))

audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=True,
)
data_path="/home/chang/bighard/AI/tts/dataset/kss/wavs/"
if colab:
    data_path="/content/drive/MyDrive/tts/dataset/kss/4"

config = HifiganConfig(
    audio=audio_config,
    batch_size=16,
    eval_batch_size=8,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=data_path,
    output_path=output_path,
    steps_to_start_discriminator=10000,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
