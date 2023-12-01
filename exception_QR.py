import cv2
import os
from pyzbar.pyzbar import decode

def read_qr_code(image_path, box):
    try:
        # Read the image
        image = cv2.imread(image_path)
        # Define the coordinates [x_min, y_min, x_max, y_max]
        [coordinates] = box
        # Extract the region defined by the coordinates
        x_min, y_min, x_max, y_max = [int(coord) for coord in coordinates]
        roi = image[y_min:y_max, x_min:x_max] 
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()       
        # Convert the image to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Decode QR codes
        qr_codes = decode(gray)
        # Print decoded text
        for qr_code in qr_codes:
            data = qr_code.data.decode('utf-8')
            new_data = data[12:25]
            print(f"The back QR Code ID card Data number is: {new_data}")
        if new_data is not None:
            # Extract filename from the original path
            _, filename = os.path.split(image_path)
            # Save the image with the original filename
            new_image_path = f'./data-saved/Back/{filename}'
            cv2.imwrite(new_image_path, roi)
            #print(f"Back Image saved as {new_image_path}")'''
        return new_data

    except Exception as e:
        print(f"An error occurred Image is not clear: {e}")
        return None
#==================================================main=======================================================
if __name__ == "__main__":
    image_path = './pictures/test07.png'  
    # Read QR code and save the image
    result = read_qr_code(image_path)

    if result is not None:
        # Handle the result as needed
        pass
    else:
        # Handle the case when an error occurred
        pass
