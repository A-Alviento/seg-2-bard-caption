# Seg2Bard-Caption

This project combines the capabilities of the Segment Anything Model (SAM), Blip-2, and Bard to generate accurate and detailed descriptions of images.

## Getting Started

This project has been tested and run on an M1 Macbook using Python 3.8. Please ensure your system meets these requirements before proceeding.

### Prerequisites and Installation

1. Start by cloning the repository:
```
git clone https://github.com/A-Alviento/seg-2-bard-caption
cd seg-2-bard-caption
pip install -e .
```
2. Next, install the required packages:
```
pip install torch torchvision torchaudio
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install eva-decord
pip install salesforce-lavis
pip install bardapi
```
*Note*: 
- For Apple Silicon machines, use `pip install torch torchvision torchaudio`. For other machines, refer to the [PyTorch official guide](https://pytorch.org/get-started/locally/) for compatible versions.
- The `pip install eva-decord` command is specific to Apple Silicon machines. Do not run this command on other types of machines.

3. To finish up, visit [Bard](https://bard.google.com/), press F12 to open the console, navigate to Session -> Application -> Cookies, and copy the value of the __Secure-1PSID cookie.

## Running the Program

1. Run the main Python script:
```
python seg-2-bard-caption.py
```
2. When prompted, enter the path to the image you want to describe.
3. Enter your Bard Key when prompted.

## Acknowledgements

This project was inspired and made possible by the following sources:
- [Segment Anything Model (SAM) by Facebook Research](https://github.com/facebookresearch/segment-anything/blob/main/README.md)
- [Caption Anything Project](https://github.com/ttengwang/Caption-Anything)
- [Bard API Repository](https://github.com/dsdanielpark/Bard-API)
- [Bard by Google](https://bard.google.com/)
- [Blip-2 Documentation on Hugging Face](https://huggingface.co/docs/transformers/main/model_doc/blip-2)