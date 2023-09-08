# Hand Gesture Recognition Deep Learning model
---
### PRODIGY_ML_Hand-gesture recognition
### Hugging Face <a href="https://huggingface.co/spaces/Harsh-Jadhav/hand_gesture-recognition" >web app</a>

Repo contains **two** modeling files
- Fastai modeling using pre-trained residual network 34 layers
- Tensorflow model with 6 layers and 1.7m params

Training data with approximately 16k images of 10 different classes into 10 different directories

Validation data with images different from training and close to real-life / expected input images

Training images are grayscale with most pixel intensity at the "hands" in the images, i.e highly processed images for training which may very well overfit and lose out on generalization in real life input images,

No transformation like grayscaling and pixel transformation are applied on input and validation images.
