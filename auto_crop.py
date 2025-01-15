from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Parameters
face_model_path = Path('yolo_weights/yolov11l-face.pt')  # Path to face detection weights
person_model_path = Path('yolo_weights/yolov8n-person.pt')  # Path to person detection weights
input_directory = Path('../data/nk-dataset')  # Directory with input images
output_directory = Path('../data/nk-dataset-crops')  # Directory for saving processed images
desired_size = 1024  # Desired output resolution
face_padding = 1.15  # Padding around face bounding boxes
person_padding = 1.1  # Padding around person bounding boxes

# Load models
face_model = YOLO(face_model_path)
person_model = YOLO(person_model_path)

def process_image(image_path, model, output_dir, padding, counter):
    # Load image
    image = cv2.imread(str(image_path))
    image_height, image_width = image.shape[:2]

    # Perform object detection
    results = model(image)

    for result in results:
        # Extract bounding boxes
        for box in result.boxes:
            # Convert tensor coordinates to integers
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_width = x2 - x1
            box_height = y2 - y1

            # Calculate padding based on percentage
            padding_x = int(box_width * (padding - 1))
            padding_y = int(box_height * (padding - 1))

            # Add padding on all sides
            x1 = int(x1) - padding_x
            y1 = int(y1) - padding_y
            x2 = int(x2) + padding_x
            y2 = int(y2) + padding_y

            # Make bounding box square
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width > box_height:
                diff = box_width - box_height
                y1 -= diff // 2
                y2 += diff // 2
            else:
                diff = box_height - box_width
                x1 -= diff // 2
                x2 += diff // 2

            # Ensure coordinates are within the image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_width, x2)
            y2 = min(image_height, y2)

            # Calculate new square size
            square_size = max(x2 - x1, y2 - y1)

            # Create a new square image with zero padding
            cropped_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)

            # Calculate offsets to place the cropped region in the center
            start_y = max(0, y1)
            start_x = max(0, x1)
            end_y = min(image_height, y2)
            end_x = min(image_width, x2)

            crop_width = end_x - start_x
            crop_height = end_y - start_y

            y_offset = (square_size - crop_height) // 2
            x_offset = (square_size - crop_width) // 2

            cropped_img[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width] = image[start_y:end_y, start_x:end_x]

            # Resize to desired square size
            resized_img = cv2.resize(cropped_img, (desired_size, desired_size))

            # Save the resized image
            ext = image_path.suffix  # Get the original file extension
            output_path = output_dir / f"{counter}{ext}"
            cv2.imwrite(str(output_path), resized_img)
            counter += 1

    return counter

def main():
    # Create output directories if they don't exist
    output_directory.mkdir(parents=True, exist_ok=True)

    # Iterate through images in the input directory
    counter = 0
    for image_path in input_directory.iterdir():
        if image_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            # Process image for face detection
            counter = process_image(image_path, face_model, output_directory, face_padding, counter)

            # Process image for person detection
            counter = process_image(image_path, person_model, output_directory, person_padding, counter)

if __name__ == '__main__':
    main()