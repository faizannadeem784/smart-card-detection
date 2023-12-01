import cv2
import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
# Set the environment variable to avoid the OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import easyocr
def divide_and_ocr(roii, language='en'):
    try:
        roi = adjust_brightness_contrast(roii, alpha=1.0, beta=0)
        # Read the image
        image = roi
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Get the height and width of the image
        height, width, _ = image.shape
        # Crop and Divide the image at the midpoint of its height
        upper_half = image[:height // 2, :]
        lower_half = image[height // 2:, :]
        # Perform OCR on the upper and lower halves
        upper_text = perform_ocr1(upper_half, language)
        lower_text, number = perform_ocr2(lower_half, language)
        return upper_text, lower_text, number

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def perform_ocr1(image_array, language='en'):
    try:
        # Convert image to RGB (EasyOCR requires RGB format)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # Create an EasyOCR reader for the specified language
        reader = easyocr.Reader([language])
        # Read text from the image
        result = reader.readtext(image_rgb)
        # Default list of texts to ignore
        text_to_ignore = ["Dj", "PAKISTAN", "National Identity Card",
                        "ISLAMIC REPUBLIC OF PAKISTAN","Name","Efulai",
                        "Father Name","Gender","Country of Stay","Pakistan",
                        "Identity Number","Date of Birth","04727","Date of Issue",
                        "Date of Expiry","Holder $","Signature","Signature","F1s",
                        "Eiulai","Holder $ Signature","E:ulai","F1si","Fl","Countr onstdi"
                        ,"M","Date 0f Birth","Holder"]
        # Filter out unwanted text and remove dashes
        filtered_results = [(detection[0], detection[1].replace('-', '')) for detection in result if detection[1] not in text_to_ignore]
        # Create a new variable to store the filtered results
        filtered_data = [detection[1] for detection in filtered_results]
        return filtered_data

    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return None

def perform_ocr2(image_array, language='en'):
    try:
        # Convert image to RGB (EasyOCR requires RGB format)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # Create an EasyOCR reader for the specified language
        reader = easyocr.Reader([language])
        # Read text from the image
        result = reader.readtext(image_rgb)
        # Default list of texts to ignore
        text_to_ignore = ["Dj", "PAKISTAN", "National Identity Card",
                        "ISLAMIC REPUBLIC OF PAKISTAN","Name","Efulai",
                        "Father Name","Gender","Country of Stay","Pakistan",
                        "Identity Number","Date of Birth","04727","Date of Issue",
                        "Date of Expiry","Holder $","Signature","Signature","F1s",
                        "Eiulai","Holder $ Signature","E:ulai","F1si","Fl","Countr onstdi"
                        ,"M","Date 0f Birth","Holder"]

        # Filter out unwanted text and remove dashes
        filtered_results = [(detection[0], detection[1].replace('-', '')) for detection in result if detection[1] not in text_to_ignore]
        # Create a new variable to store the filtered results
        filtered_data = [detection[1] for detection in filtered_results]
        # Extract text from the result
        # Extract and print only valid numbers
        extracted_numbers = [detection[1] for detection in filtered_results if detection[1].isdigit()]
        # Print the extracted numbers
        if extracted_numbers is not None:
            for number in extracted_numbers:
                print('This is the front ID card number:', number)        
        return filtered_data, number

    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return None

def OCR_function(image_path, language, box):
    # Load your image
    image = cv2.imread(image_path)
    # Define the coordinates [x_min, y_min, x_max, y_max]
    [coordinates] = box

    # Extract the region defined by the coordinates
    x_min, y_min, x_max, y_max = [int(coord) for coord in coordinates]
    roi = image[y_min:y_max, x_min:x_max]
    # Call the divide_and_ocr function to separate and perform OCR on image regions
    upper_text, lower_text, number = divide_and_ocr(roi, language)
    # Extract specific fields from the OCR results for upper and lower halves
    filtered_data1 = upper_text
    new_filtered_data1 = {'Name': filtered_data1[0], 'Father Name': filtered_data1[1]}

    filtered_data2 = lower_text
    new_filtered_data2 = {'Identity Card Number': filtered_data2[0],
                          'DOB': filtered_data2[1], 'DOI': filtered_data2[2], 'DOE': filtered_data2[3]}

    # Combine and extend the extracted data from both upper and lower halves
    filtered_data1.extend(filtered_data2)
    # Merge dictionaries for a unified result
    merged_dict = {**new_filtered_data1, **new_filtered_data2}
    #print(merged_dict)
    # Return the merged dictionary, the combined list, and additional extracted numbers
    
    # Extract filename from the original path
    _, filename = os.path.split(image_path)
    # Save the image with the original filename
    new_image_path = f'./data-saved/Front/{filename}'
    cv2.imwrite(new_image_path, roi)
    #print(f"Back Image saved as {new_image_path}")
    print(merged_dict)
    return merged_dict, filtered_data1, number
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
# ========================================main================================================================
if __name__ == "__main__":
    # Provide the path to your image
    image_path = "./pictures/test06.png"
    # Specify the language (default is English 'en')
    language = 'en'
    dict, list, num = OCR_function(image_path,language)
    print("Dicitionary:",dict)
    print("List:",list)
    print("Number:",num)