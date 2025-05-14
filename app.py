import streamlit as st
from ultralytics import YOLO
import os
import csv
from PIL import Image

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Cost estimation logic
COST_MAPPING = {
    'Dent': 1200,
    'Glass_Break': 3000,
    'Scratch': 800
}

# Streamlit Interface
st.title("Car Damage Detection & Cost Estimation")

st.write("""
    Upload an image of a car, and we'll detect any damages and provide a cost estimate for repairs.
""")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Load image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Save uploaded image for processing
    image_path = "temp_uploaded_image.jpg"
    img.save(image_path)

    # Run YOLO model on the uploaded image
    results = model(image_path)

    # Get detected classes
    detected_classes = results[0].boxes.cls.cpu().tolist()
    class_names = model.names
    detected_labels = [class_names[int(cls_id)] for cls_id in detected_classes]
    total_cost = sum([COST_MAPPING.get(label, 0) for label in detected_labels])

    # Show detected damages and cost
    st.subheader("Detected Damages:")
    st.write(", ".join(detected_labels))

    st.subheader(f"Estimated Repair Cost: ₹{total_cost}")

    # Show result image with bounding boxes
    result_img = results[0].plot()
    st.image(result_img, caption="Annotated Image with Detected Damages", use_column_width=True)

    # Provide download link for repair report (CSV format)
    report_filename = "damage_report.csv"
    with open(report_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Detected_Damages", "Estimated_Cost"])
        writer.writerow([image_path, ", ".join(detected_labels), total_cost])

    st.download_button(
        label="Download Repair Report",
        data=open(report_filename, "rb").read(),
        file_name=report_filename,
        mime="text/csv"
    )

