import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def crop_document(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Blurring
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('Blurred', blurred)
    cv2.waitKey(0)

    # Step 2: Edge detection
    edged = cv2.Canny(blurred, 75, 200)
    cv2.imshow('Edged', edged)
    cv2.waitKey(0)

    # Step 3: Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 1)
    cv2.imshow('All Contours', image_contours)
    cv2.waitKey(0)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = 10000  # Minimum area threshold to ignore small rectangles
    document_contour = None

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)  # Increased approximation accuracy

        if len(approx) == 4:
            document_contour = approx
            break

    if document_contour is None:
        print("No document found")
        return None

    # Draw the found contour on the image
    cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 3)
    cv2.imshow('Document Contour', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    warped = four_point_transform(image, document_contour.reshape(4, 2))

    return warped


# Usage
image_path = '142359.jpg'
cropped_image = crop_document(image_path)
if cropped_image is not None:
    cv2.imshow('Cropped Document', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to crop the document.")
