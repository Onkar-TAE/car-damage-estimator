from ultralytics import YOLO
import os
import csv

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Folder with test images
image_folder = r"D:\New Start\test\images"
output_folder = r"D:\New Start\test\results"
os.makedirs(output_folder, exist_ok=True)

# CSV file path
csv_path = os.path.join(output_folder, "damage_report.csv")

# Cost estimation logic
COST_MAPPING = {
    'Dent': 1200,
    'Glass_Break': 3000,
    'Scratch': 800
}

# Open CSV file for writing
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Detected_Damages", "Estimated_Cost"])

    # Loop through all images
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            results = model(image_path)

            # Get classes
            detected_classes = results[0].boxes.cls.cpu().tolist()
            class_names = model.names
            detected_labels = [class_names[int(cls_id)] for cls_id in detected_classes]
            total_cost = sum([COST_MAPPING.get(label, 0) for label in detected_labels])

            # Print info
            print(f"\nImage: {filename}")
            print(f"Detected: {detected_labels}")
            print(f"Estimated Repair Cost: ₹{total_cost}")

            # Save annotated image
            result_img_path = os.path.join(output_folder, f"result_{filename}")
            results[0].save(filename=result_img_path)

            # Write to CSV
            writer.writerow([filename, ", ".join(detected_labels), total_cost])

print(f"\n✅ CSV report saved at: {csv_path}")