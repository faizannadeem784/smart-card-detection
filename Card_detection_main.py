import cv2
import numpy as np
import csv
import os
from yolov8 import YOLOv8
from new_front import OCR_function  
from exception_QR import read_qr_code
from mtcnn import MTCNN
# Initialize yolov8 object detector
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

#Functions================================================
def save_data_to_csv(data, csv_file_path="data.csv"):
    try:
        # Try to open the file in read mode to check if it's empty
        with open(csv_file_path, mode='r') as file:
            is_empty = file.read().strip() == ''
    except FileNotFoundError:
        # If the file doesn't exist, consider it as empty
        is_empty = True
    # Open the file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        # Create a CSV writer object
        csv_writer = csv.writer(file)
        if is_empty:
            # Write header if the file is empty
            csv_writer.writerow(["Name", "Father Name", "ID number", "DOB", "DOI", "DOE"])
        # Write the data to the CSV file
        csv_writer.writerow(data)
    print("Data saved to", csv_file_path)

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    # Apply brightness and contrast adjustment using cv2.convertScaleAbs
    # The formula is: output = alpha * input + beta
    # alpha controls contrast, beta controls brightness
    # ConvertScaleAbs applies the following operation on each pixel:
    # output_pixel = saturate(alpha * input_pixel + beta)
    # saturate ensures that the pixel values are within the valid range (0 to 255)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # Return the adjusted image
    return adjusted

def make_predictions(img):
    # Get bounding boxes, scores, and class IDs using YOLOv8 detector
    boxes, scores, class_ids = yolov8_detector(img)
    # Draw the detections on the image
    combined_img = yolov8_detector.draw_detections(img)
    # Adjust brightness and contrast of the image
    adjusted_img = adjust_brightness_contrast(combined_img, alpha=1.2, beta=10)
    # Create a named window for displaying the detected objects
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    # Display the adjusted image with detected objects
    cv2.imshow("Detected Objects", adjusted_img)
    # Wait for a key press (0 means wait indefinitely)
    cv2.waitKey(0)
    # Return the detected bounding boxes
    return boxes

def face_detection(roi):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier('./models/data.xml')
    # Convert the region of interest (ROI) to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale i   mage using the face cascade
    # scaleFactor and minNeighbors are parameters affecting the detection sensitivity
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # Return the detected faces as bounding boxes (x, y, width, height)
    return faces

def face_detection_mtcnn(roi):
    # Load the MTCNN face detection model
    detector = MTCNN()
    # Detect faces in the region of interest (ROI)
    faces_info = detector.detect_faces(roi)

    # Extract bounding boxes from the detected faces
    bounding_boxes = [info['box'] for info in faces_info]

    return bounding_boxes

def save_img_path(image_path,roi):
    # Extract filename from the original path
    _, filename = os.path.split(image_path)
    # Save the image with the original filename
    new_image_path = f'./data-saved/Back/{filename}'
    cv2.imwrite(new_image_path, roi)
    print(f"Image saved as {new_image_path}")

def read_box(box, path, back_path,box2):
    pic = box
    pic2=box2
    # Specify the language for OCR
    language = 'en'
    # Path to the front side of the ID card image
    image_path = path
    # Convert the list of box coordinates to a list of tuples
    boxes = [(x, y, w, h) for x, y, w, h in box]
    # Read the image from the specified path
    image = cv2.imread(image_path)
     # Iterate over each bounding box and extract the region of interest (ROI)
    for box in boxes:
        # Extract box coordinates
        x, y, w, h = map(int, box)  
        # Calculate the midpoint of the box
        mid_x = x + w // 2
        # Extract the right and left sides of the box as separate ROIs
        right_side_roi = image[y:y+h, mid_x:x+w]
        left_side_roi = image[y:y+h, x:mid_x]
        # Perform face detection on the right and left sides
        right_faces = face_detection(right_side_roi)
        left_faces = face_detection(left_side_roi)

        # Check if faces are detected on the right side
        if len(right_faces) > 0:
            print("Picture found, this is a new CNIC card")
            # Perform OCR on the front side of the card
            dict, data_list, front = OCR_function(image_path, language, pic)
             # Read QR code from the back side of the card
            back = read_qr_code(back_path,pic2)
            # Verify if the front and back of the card match
            if front == back:
                print("Front and Back Card are verified\nThe Extracted data is:")
                print(dict)
                print("Front and back pictures are saved in [data-saved directory]")
                save_data_to_csv(data_list)
            else:
                print('ID Cards do not match. Please place the same ID Card.')
        # Check if faces are detected on the left side
        elif len(left_faces) > 0:
            print("This is an old ID card")
            # Perform OCR on the front side of the card
            dict, data_list, front = OCR_function(image_path, language, pic)
            # Read QR code from the back side of the card
            back = read_qr_code(back_path, pic2)
            # Verify if the front and back of the card match
            if front == back:
                print("Front and Back Card are verified\nThe Extracted data is:")
                print(dict)
                print("Front and back pictures are saved in [data-saved directory]")
                save_data_to_csv(data_list)
            else:
                print('ID Cards do not match. Please place the same ID Card.')
        else:
            print("Not found. This is not an ID card.")

##############################################################################################
if __name__ == "__main__":
    # Path to the front side of the ID card image
    path = 'pictures/imageid.jpeg'
    #path = 'pictures/test06.png'
    # Path to the back side of the ID card image
    back_path = './pictures/test07.png'
    # Read the front side image
    img = cv2.imread(path)
    img2 = cv2.imread(back_path)
    # Make predictions using an object detection model (assuming make_predictions returns bounding boxes)
    box = make_predictions(img)
    box2 = make_predictions(img2)
    # Process the detected bounding boxes and perform further actions
    read_box(box, path, back_path,box2)


