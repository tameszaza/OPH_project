from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from ultralytics import YOLO
import shutil
import cv2
import numpy as np
import uuid
from pdf2image import convert_from_path
from PIL import Image

app = Flask(__name__)

index_table = (0,9,17, 25, 36, 46, 56, 67, 75, 90, 95, 103)

subjects = (
    ("ค30203", "หลักคณิตศาสตร์"),
    ("ค30204", "สถิติ"),
    ("ค30205", "แคลคูลัสขั้นสูง"),
    ("ค30206", "ทฤษฎีกราฟ"),
    ("ค30207", "กระบวนการสโตแคสติก"),
    ("ค30208", "พีชคณิตเชิงเส้นเบื้องต้น"),
    ("ค30209", "การวิเคราะห์เชิงจริงเบื้องต้น"),
    ("ค30210", "สมการเชิงอนุพันธ์สามัญ"),
    ("ค30211", "การหาค่าเหมาะสมและการเรียนรู้แบบเสริมกำลัง"),
    ("ค30212", "คณิตศาสตร์เชิงคำนวณและการสร้างแบบจำลอง"),
    ("ค30213", "หัวข้อพิเศษทางคณิตศาสตร์"),
    ("ว30301", "กลศาสตร์คลาสสิก"),
    ("ว30302", "ทัศนศาสตร์ประยุกต์"),
    ("ว30303", "ฟิสิกส์ดาราศาสตร์เบื้องต้น"),
    ("ว30304", "ทฤษฎีสนามคลาสสิกและฟิสิกส์อนุภาคเบื้องต้น"),
    ("ว30305", "กลศาสตร์สถิติ"),
    ("ว30306", "กลศาสตร์ฐานแคลคูลัส"),
    ("ว30307", "ไฟฟ้าและแม่เหล็กฐานแคลคูลัส"),
    ("ว30308", "หัวข้อพิเศษทางฟิสิกส์"),
    ("ว30401", "เคมีในชีวิตประจำวัน"),
    ("ว30402", "แนวหน้าของวัสดุศาสตร์ชั้นนำ"),
    ("ว30403", "เคมีวิเคราะห์เชิงไฟฟ้าเบื้องต้น"),
    ("ว30404", "เทคนิคเคมีวิเคราะห์ขั้นสูง"),
    ("ว30405", "สมบัติของสสารขั้นสูง"),
    ("ว30406", "เคมีอนินทรีย์ จลนศาสตร์เคมีขั้นสูง และอุณหพลศาสตร์เคมี"),
    ("ว30407", "เคมีอินทรีย์ขั้นสูง"),
    ("ว30408", "กลศาสตร์ควอนตัมเชิงโมเลกุลเบื้องต้นและสเปกโทรสโกปี"),
    ("ว30409", "การวิเคราะห์และเทคนิคทางชีวเคมี"),
    ("ว30410", "หัวข้อพิเศษทางเคมี"),
    ("ว30501", "ปฏิบัติการชีววิทยาสังเคราะห์ขั้นพื้นฐาน"),
    ("ว30502", "ปฏิบัติการชีววิทยาของพืชและสัตว์ขั้นสูง"),
    ("ว30503", "เทคนิคปฏิบัติการทางจุลชีววิทยา"),
    ("ว30504", "เทคโนโลยีชีวภาพ"),
    ("ว30505", "พันธุศาสตร์ขั้นสูงและชีววิทยาโมเลกุล"),
    ("ว30506", "ประสาทชีววิทยา"),
    ("ว30507", "ชีววิทยาของมะเร็ง"),
    ("ว30508", "ชีววิทยาเพื่อสิ่งแวดล้อมที่ยั่งยืน"),
    ("ว30509", "นิเวศวิทยาภาคสนาม"),
    ("ว30510", "หัวข้อพิเศษทางชีววิทยา"),
    ("ว30601", "บรรพชีวินวิทยา"),
    ("ว30602", "อัญมณีวิทยา"),
    ("ว30603", "หัวข้อพิเศษทางวิทยาศาสตร์โลก"),
    ("ว30604", "หัวข้อพิเศษทางดาราศาสตร์"),
    ("ว30703", "การเขียนโปรแกรมคอมพิวเตอร์เบื้องต้น 2"),
    ("ว30704", "โครงสร้างข้อมูลและอัลกอริทึมเบื้องต้น"),
    ("ว30705", "คอมพิวเตอร์ออกแบบและสร้างต้นแบบ 1"),
    ("ว30706", "คอมพิวเตอร์ออกแบบและสร้างต้นแบบ 2"),
    ("ว30707", "การเขียนโปรแกรมภาษาไพทอน"),
    ("ว30708", "วิทยาการคอมพิวเตอร์ทั่วไป 1"),
    ("ว30709", "วิทยาการคอมพิวเตอร์ทั่วไป 2"),
    ("ว30710", "การเรียนรู้ของเครื่อง 1"),
    ("ว30711", "การเรียนรู้ของเครื่อง 2"),
    ("ว30712", "ระบบปฏิบัติการหุ่นยนต์และการประยุกต์"),
    ("ว30713", "หัวข้อพิเศษทางวิทยาการคอมพิวเตอร์"),
    ("ว30807", "พื้นฐานการคำนวณและการจำลองเชิงวิทยาศาสตร์"),
    ("ว30808", "ฟิสิกส์เชิงคำนวณเบื้องต้น"),
    ("ว30809", "ฟิสิกส์เชิงคำนวณขั้นสูง"),
    ("ว30810", "เคมีเชิงคำนวณพื้นฐาน"),
    ("ว30811", "ชีวสารสนเทศและชีววิทยาเชิงคำนวณเบื้องต้น"),
    ("ว30812", "คณิตศาสตร์สำหรับฟิสิกส์"),
    ("ว30813", "ชีวฟิสิกส์"),
    ("ว30814", "วิศวกรรมโปรตีนและการนำไปใช้"),
    ("ว30815", "นวัตกรรมเพื่อสิ่งแวดล้อมที่ยั่งยืน"),
    ("ว30816", "ภูมิศาสตร์สิ่งแวดล้อม"),
    ("ว30817", "การท่องเที่ยวอย่างยั่งยืน"),
    ("ว30901", "ปรัชญาและประวัติศาสตร์วิทยาศาสตร์"),
    ("ว30902", "ทรัพย์สินทางปัญญา"),
    ("ท30201", "การเล่าเรื่อง"),
    ("ท30202", "การเขียนสารคดี"),
    ("ท30203", "ความพิศวงในวรรณกรรม"),
    ("ท30204", "รสแห่งวรรณกรรม"),
    ("ท30205", "วรรณกรรมพื้นบ้าน"),
    ("ท30206", "วรรณกรรมวิทยาศาสตร์"),
    ("ท30207", "คติชนวิทยา"),
    ("ท30208", "เอกัตศึกษาภาษาและวรรณกรรม"),
    ("ศ30201", "พื้นฐานการออกแบบ"),
    ("ศ30202", "ศิลปะและเทคโนโลยี"),
    ("ศ30203", "จิตรกรรมสร้างสรรค์"),
    ("ศ30204", "ศิลปะเพื่อสังคม"),
    ("ศ30205", "ทักษะดนตรีไทย"),
    ("อ30207", "การพูดในที่ชุมชน"),
    ("อ30208", "วรรณกรรมมีชีวิต"),
    ("อ30209", "การสื่อสารระหว่างวัฒนธรรม"),
    ("อ30210", "การสื่อสารทางวิทยาศาสตร์"),
    ("อ30211", "กลยุทธ์การสอบเพื่อเตรียมตัวสำหรับการสอบแบบทดสอบมาตรฐาน"),
    ("ส30201", "จิตวิทยาการดำรงชีวิต"),
    ("ส30202", "มหาศึกชิงบัลลังก์กับสังคมศึกษา"),
    ("ส30203", "เอกัตศึกษา สังคมศาสตร์และมนุษยศาสตร์ 1"),
    ("ส30204", "เอกัตศึกษา สังคมศาสตร์และมนุษยศาสตร์ 2"),
    ("ส30205", "สัมมนาประวัติศาสตร์"),
    ("อ30207", "การพูดในที่ชุมชน"),
    ("อ30208", "วรรณกรรมมีชีวิต"),
    ("อ30209", "การสื่อสารระหว่างวัฒนธรรม"),
    ("อ30210", "การสื่อสารทางวิทยาศาสตร์"),
    ("อ30211", "กลยุทธ์การสอบเพื่อเตรียมตัวสำหรับการสอบแบบทดสอบมาตรฐาน"),
    ("ก30201", "ภาษาเกาหลี 1"),
    ("ก30202", "ภาษาเกาหลี 2"),
    ("จ30201", "ภาษาจีน 1"),
    ("จ30202", "ภาษาจีน 2"),
    ("ซ30201", "ภาษารัสเซีย 1"),
    ("ซ30202", "ภาษารัสเซีย 2"),
    ("ญ30201", "ภาษาญี่ปุ่น 1"),
    ("ญ30202", "ภาษาญี่ปุ่น 2"),
    ("ป30201", "ภาษาสเปน 1"),
    ("ป30202", "ภาษาสเปน 2"),
    ("ฝ30201", "ภาษาฝรั่งเศส 1"),
    ("ฝ30202", "ภาษาฝรั่งเศส 2"),
    ("ย30201", "ภาษาเยอรมัน 1"),
    ("ย30202", "ภาษาเยอรมัน 2"),
    ("ร30201", "ภาษาอาหรับ 1"),
    ("ร30202", "ภาษาอาหรับ 2")
)
model = YOLO('best8.pt')
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
def crop_document_image(filepath, save_path):
    # Read the image from the file path
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError("Image not found at the specified path.")
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection with adjusted thresholds to reduce sensitivity to small edges
    edged = cv2.Canny(blurred, 100, 248)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours based on area
    min_contour_area = 1000  # Minimum contour area to consider
    large_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    
    if not large_contours:
        raise ValueError("No large contours found.")
    
    # Find the largest contour by area
    largest_contour = max(large_contours, key=cv2.contourArea)
    
    # Approximate the largest contour
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        screen_contour = approx
    else:
        screen_contour = largest_contour
    
    # Apply the perspective transformation
    warped = four_point_transform(image, screen_contour.reshape(4, 2))
    
    # Save the cropped image
    cv2.imwrite(save_path, warped)
    
    return warped

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Construct the destination points to use for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    output = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return output

def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect

def annotate_image(img, detections, code_boxes, sticker_boxes, subject_list, filename):
    for bbox, score, color in sticker_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{color} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for bbox, score in code_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'Code {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for subject in subject_list:
        color, (code, name) = subject
        cv2.putText(img, f'{color} {code} {name}', (10, 30 * (subject_list.index(subject) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    output_path = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")


def preprocess(photo_path):
    crop_document_image(photo_path, os.path.join(UPLOAD_FOLDER, 'test.png'))

def annotate_image(img, detections, code_boxes, sticker_boxes, subject_list, filename):
    for bbox, score, color in sticker_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{color} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for bbox, score in code_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'Code {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for subject in subject_list:
        color, (code, name) = subject
        cv2.putText(img, f'{color} {code} {name}', (10, 30 * (subject_list.index(subject) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    output_path = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")

def process_image(image_path):
    img_list = preprocess(image_path)
    subject_list = []
    for i, (filename, element) in enumerate(img_list):
        subject_list = get_selected_choices(i, element, subject_list, filename)
    print(subject_list)
    Export_Data_To_Sheets(subject_list)
    return subject_list

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = r'D:\coding\OPH\OPH_project\Adk\Sheet\oph2024-431003-672c55d29331.json'
SAMPLE_SPREADSHEET_ID_input = '1hVjR1LXUF7OuYxyi40DYwskHD97ChhiHkmttSFnBqQc'
SAMPLE_RANGE_NAME = 'A1:GG1000'

def SheetData():
    global values_input, service, df
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = build('sheets', 'v4', credentials=credentials)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result_input = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID_input,
                                      range=SAMPLE_RANGE_NAME).execute()
    values_input = result_input.get('values', [])

    if not values_input:
        print('No data found.')
        return

    # Debugging information
    print("Values Input:", values_input)
    print("Length of Values Input:", len(values_input))

    # Ensure there is at least one row of data
    if len(values_input) < 1:
        print("No data in the sheet.")
        return

    # Ensure there is a header row
    if len(values_input[0]) < 1:
        print("No header row in the sheet.")
        return

    # Strip any extra whitespace from the column names
    columns = [col.strip() for col in values_input[0]]
    df = pd.DataFrame(values_input[1:], columns=columns)
    return df

def EditData(df,list):
    color_map = {
    "Pink": 3,
    "Blue": 4,
    "Orange": 5,
    "Brown": 6
    }
    Sem3=[]
    Sem4=[]
    Sem5=[]
    Sem6=[]
    sem_dict = {
    3: Sem3,
    4: Sem4,
    5: Sem5,
    6: Sem6
    }

    new_row = {col: 0 for col in df.columns}
    for outer_key, (inner_value1, inner_value2) in list:
        new_row[inner_value1]=color_map[outer_key]
        chosen_list=sem_dict.get(color_map[outer_key],[])
        chosen_list.append(inner_value1)
    str3 = ','.join(Sem3)
    str4 = ','.join(Sem4)
    str5 = ','.join(Sem5)
    str6 = ','.join(Sem6)
    new_row['sem3']=str3
    new_row['sem4']=str4
    new_row['sem5']=str5
    new_row['sem6']=str6
    df=df.append(new_row,ignore_index=True)
    

    df=df.append(new_row,ignore_index=True)
    return df
        
    

def Export_Data_To_Sheets(sbj_ls):
    df=SheetData()
    df2=EditData(df,list)
    response_date = service.spreadsheets().values().update(
        spreadsheetId=SAMPLE_SPREADSHEET_ID_input,
        valueInputOption='RAW',
        range=SAMPLE_RANGE_NAME,
        body=dict(
            majorDimension='ROWS',
            values=df2.T.reset_index().T.values.tolist())
    ).execute()
    print('Sheet successfully Updated')



def get_selected_choices(i, img, subject_list, filename):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load your trained model
    model = YOLO('best32.pt')

    # Run the detection
    results = model(img_rgb)

    # Parse results (handling it as a list)
    detections = results[0].boxes  # Assuming the first item in results contains our detections

    code_boxes = []
    sticker_boxes = []

    img_area = img.shape[0] * img.shape[1]
    max_area = 0.05 * img_area

    # Process each detection
    for det in detections:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        score = det.conf.item()
        class_id = int(det.cls.item())

        bbox = [x1, y1, x2, y2]
        bbox_area = (x2 - x1) * (y2 - y1)

        # Assign class names and categorize detections
        if class_id == 4:  # Code
            if bbox_area <= max_area:
                code_boxes.append((bbox, score))
            else:
                print(f"Dropping code box larger than 5% of image area: {bbox}")
        else:
            class_names = ['Blue', 'Orange', 'Pink', 'Brown']
            if class_id in range(len(class_names)):
                sticker_boxes.append((bbox, score, class_names[class_id]))

    # Match stickers to the nearest code box
    for sticker_bbox, _, color in sticker_boxes:
        nearest_box = None
        min_distance = float('inf')

        for code_bbox, _ in code_boxes:
            # Calculate distance between the centers of the sticker and code box
            distance = np.linalg.norm(np.array([
                (sticker_bbox[0] + sticker_bbox[2]) / 2 - (code_bbox[0] + code_bbox[2]) / 2,
                (sticker_bbox[1] + sticker_bbox[3]) / 2 - (code_bbox[1] + code_bbox[3]) / 2
            ]))

            if distance < min_distance:
                min_distance = distance
                nearest_box = code_bbox

        # Determine the order of the code box selected by the sticker
        sorted_codes = sorted(code_boxes, key=lambda x: x[0][1])  # Sort by vertical position

        # Check if the nearest_box is in sorted_codes
        if nearest_box in [code[0] for code in sorted_codes]:
            order = [code[0] for code in sorted_codes].index(nearest_box) + 1  # Get order based on sorted position
            print(order)
            subject_list.append([color, subjects[index_table[i] + order - 1]])

    # Annotate image for debugging
    annotate_image(img, detections, code_boxes, sticker_boxes, subject_list, filename)
    
    return subject_list

if __name__ == '__main__':
    process_image('142359.jpg')