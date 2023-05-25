# seg-2-bard-caption

This projects implements Segment Anything Model (SAM), Blip-2 and Bard to provide detailed and accurate descriptions of images

## Getting Started

Note: This project was tested and ran on an M1 Macbook. Python=3.8 was used.

First, clone repository:
```
git clone https://github.com/A-Alviento/seg-2-bard-caption
cd seg-2-bard-caption
pip install -e
```

Next, install the following:
```
pip install torch torchvision torchaudio
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install eva-decord
pip install salesforce-lavis
pip install bardapi
```
Note: `pip install eva-decord` is for apple silicon. Don't run otherwise

Then download a [model checkpoint](#model-checkpoints)

