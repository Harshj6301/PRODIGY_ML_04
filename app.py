import streamlit as st
from fastai.vision.all import *

# Load the pre-trained model
model_path = "assets/model.pkl"  # Replace with the path to your model file
learn = load_learner(model_path)

# Define a function to make predictions on an image
def predict(image):
    img = PILImage.create(image)
    pred, _, probs = learn.predict(img)
    return pred, probs[pred]

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

if capture:
    st.write("Click the button to capture the image")
    capture_button = st.button("Capture")

    if capture_button:
        # Capture the image from the webcam
        # You can use Python libraries like OpenCV for this
        # Replace this with your code for capturing an image
        
        # For example:
        # cap = cv2.VideoCapture(0)
        # ret, frame = cap.read()
        # cv2.imwrite("captured_image.jpg", frame)
        # cap.release()
        
        # After capturing the image, predict and display it
        captured_image_path = "captured_image.jpg"  # Replace with the actual path
        st.image(captured_image_path, caption="Captured Image", use_column_width=True)
        
        # Make predictions
        prediction, confidence = predict(captured_image_path)
        
        # Display the prediction
        st.subheader("Prediction:")
        st.write(f"Gesture: {prediction}")
        st.write(f"Confidence: {confidence:.2f}")

# Example images
st.sidebar.title("Example Images")
example_images = {
    "Image 1": "example_image1.jpg",
    "Image 2": "example_image2.jpg",
    "Image 3": "example_image3.jpg",
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
