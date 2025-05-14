from ultralytics import YOLO

# Load a pre-trained YOLOv8n model (n = nano, small and fast)
model = YOLO("yolov8n.pt")

# Train the model on your Roboflow dataset
model.train(
    data="data.yaml",   # Make sure this path points to your Roboflow data.yaml file
    epochs=20,          # You can increase this for better performance
    imgsz=640,          # Image size for training
    batch=8             # Batch size (adjust based on your PC's RAM)
)
