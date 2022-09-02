tts --text "$1" \
 --model_path best_model.pth \
 --config_path config.json \
 --vocoder_path ../../hifigan/last_run/best_model.pth \
 --vocoder_config_path ../../hifigan/last_run/config.json \
 --out_path test.wav && \
 aplay test.wav