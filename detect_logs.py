import cv2
import numpy as np

def detect_logs(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to read the image at '{image_path}'")
        return []

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to remove small noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

    # Find contours in the processed image
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    log_dimensions = []

    for contour in contours:
        # Filter contours based on area to remove small details
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Calculate the bounding box for the contour
        x, y, w, h = cv2.boundingRect(approx)

        # Draw the bounding box on the original image (for visualization)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the diameter of the log (assuming circular shape)
        diameter = min(w, h)

        # Append the diameter to the list
        log_dimensions.append(diameter)

        # Display the measurement near the rectangle
        cv2.putText(image, f"{diameter} pixels", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the image with rectangles and measurements
    cv2.imshow("Detected Logs", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return log_dimensions

# Example usage
image_path = "rear-view-truck.jpeg"
log_dimensions = detect_logs(image_path)
print("Log Dimensions:", log_dimensions)
