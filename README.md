# sanime2x

requires python (tested under 3.14) and:

- `torch`
- `torchvision`
- `Pillow`

## training

- edit & run `main.py` for plain CNN w/ global residual link and `main_resnet.py` for ResNet.
- definitely download [sanime2x-2k](https://huggingface.co/datasets/bctnry/sanime2x-2k), unzip & change the path. the dataloader script expects the `part_{i}/{id}.jpeg` structure unchanged.
- use `dataloader` if you don't want to pre-load all the images into the memories (but are willing to take the speed penalty of decoding JPEG images on the fly)
- `enlarge.py` and `enlarge_resnet.py` are example code of how you can use the trained models.

## trained models

you're expected to train it yourself (i don't know how good the thingy you need it to be), but trained models can be found at [HuggingFase](https://huggingface.co/datasets/bctnry/sanime2x-2k).

## license

code & model is in cc0/public domain.

