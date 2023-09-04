import streamlit as st
from fastai.vision.all import *
import numpy as np
import cv2

# Load the pre-trained model
model_path = "assets/model-r34.pkl"  # Replace with the path to your model file
learn = load_learner(model_path)

# Define a function to make predictions on an image
def predict(image):
    img = PILImage.create(image)
    pred, _, probs = learn.predict(img)
    pred_idx = torch.argmax(probs)
    return pred, probs[pred_idx].item()

# Streamlit app
st.title("Hand Gesture Recognition")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    prediction, confidence = predict(uploaded_image)
    
    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"Gesture: {prediction}")
    st.write(f"Confidence: {confidence:.2f}")

# Capture an image from webcam
capture = st.checkbox("Capture an Image from Webcam")

def capture_image():
    st.write("Click the button to capture the image")
    capture_button = st.button("Capture")

    if capture_button:
        # Capture the image from the webcam
        cap = cv2.VideoCapture(0)

        num_cameras = 5  # Try increasing this number if necessary
        for i in range(num_cameras):
            cap = cv2.VideoCapture(i)
            
        if not cap.isOpened():
            print(f"Camera {i}: Not opened")
        else:
            print(f"Camera {i}: Opened")
            cap.release()
        
        if not cap.isOpened():
            st.error("Error: Unable to access the camera")
            return
        
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not capture image")
            return
        
        # After capturing the image, predict and display it
        captured_image_path = "captured_image.jpg"  
        cv2.imwrite(captured_image_path, frame)

        # Display the captured image
        st.image(cv2.imread(captured_image_path), caption = "Captured Image", use_column_width = True)
        
        # Make predictions
        prediction, confidence = predict(captured_image_path)
        
        # Display the prediction
        st.subheader("Prediction:")
        st.write(f"Gesture: {prediction}")
        st.write(f"Confidence: {confidence:.2f}")

# Capture calling
if capture:
    capture_image()

# Example images
st.sidebar.title("Example Images")
example_images = {
    "Image 1": "assets/fist.png",
    "Image 2": "assets/ok.png"
}

selected_example = st.sidebar.selectbox("Select an Example Image", list(example_images.keys()))

if selected_example:
    selected_image_path = example_images[selected_example]
    st.image(selected_image_path, caption=selected_example, use_column_width=True)
    
    # Make predictions for the example image
    example_prediction, example_confidence = predict(selected_image_path)
    
    # Display the prediction
    st.sidebar.subheader("Prediction:")
    st.sidebar.write(f"Gesture: {example_prediction}")
    st.sidebar.write(f"Confidence: {example_confidence:.2f}")
