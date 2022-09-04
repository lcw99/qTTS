if test -z "$2" 
then
    devices="0,1"
else
    devices="$2"
fi
CUDA_VISIBLE_DEVICES=$devices python -m trainer.distribute --script "$1"
