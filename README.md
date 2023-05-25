# seg-2-bard-caption

This projects implements Segment Anything Model (SAM), Blip-2 and Bard to provide detailed and accurate descriptions of images

## Getting Started
Note: This project was tested and ran on an M1 Macbook. Python=3.8 was used. <br><br>
### Prerequisites and Installations
1. First, clone repository:
```
git clone https://github.com/A-Alviento/seg-2-bard-caption
cd seg-2-bard-caption
pip install -e
```
2. Next, install the following:
```
pip install torch torchvision torchaudio
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install eva-decord
pip install salesforce-lavis
pip install bardapi
```
Note: 
- `pip install torch torchvision torchaudio` is for apple silicon. Refer to https://pytorch.org/get-started/locally/ for compatible versions
- `pip install eva-decord` is for apple silicon. Don't run otherwise <br><br>

3. Finally: 
- Visit https://bard.google.com/
- F12 for console
- Session -> Application -> Cookies -> Copy the value of __Secure-1PSID cookie

## Running the program
1. Simply run:
```
python seg-2-bard-caption.py
```
2. Enter path to desired image
3. Enter Bard Key

## Acknowledgements
https://github.com/facebookresearch/segment-anything/blob/main/README.md
https://github.com/ttengwang/Caption-Anything
https://github.com/dsdanielpark/Bard-API
https://bard.google.com/
https://huggingface.co/docs/transformers/main/model_doc/blip-2

