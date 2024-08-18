#this python file is the api 
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from ultralytics import YOLO
import shutil
import cv2
import numpy as np
import uuid
from pdf2image import convert_from_path
from PIL import Image
from ultralytics import YOLO
import time
import pandas as pd
from googleapiclient.discovery import build
from google.oauth2 import service_account
import pickle
import socket
import platform

app = Flask(__name__)

index_table = (0,9,17, 25, 36, 46, 56, 67, 75, 85, 90, 98)

class data:
    def __init__(self, stem_count, foreign_count, total, blue_count, pink_count, brown_count, orange_count, subject_list):
        self.stem_count = stem_count
        self.foreign_count = foreign_count
        self.total = total
        self.blue_count = blue_count
        self.pink_count = pink_count
        self.brown_count = brown_count
        self.orange_count = orange_count
        self.subject_list = subject_list
        
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


# Load model here
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
detected_class = set()
hostname = socket.gethostname()
system_name = platform.system()
node_name = platform.node()
release = platform.release()
version = platform.version()
machine = platform.machine()
processor = platform.processor()

# add preprocessing function here input parameter

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({'message': 'Test route reached'}), 200


def validate_output(output):
    reasons = []
    is_valid = True
    
    if output.stem_count < 8:
        is_valid = False
        reasons.append('Stem count is less than 8.')
    
    if output.foreign_count != 2:
        is_valid = False
        reasons.append('Foreign count is not equal to 2.')
    
    if output.total < 14:
        is_valid = False
        reasons.append('Total count is less than 14.')
    
    if any([output.blue_count > 4, output.pink_count > 4, output.brown_count > 4, output.orange_count > 4]):
        is_valid = False
        reasons.append('One or more color counts are greater than 4.')
    
    return is_valid, reasons

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print('No file part')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_filename = timestamp + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Assuming process_image returns an instance of Data
        output = process_image(filepath)
        is_valid, reasons = validate_output(output)
        
        response = {
            'validity': is_valid,
            'reasons': reasons if not is_valid else None,
            'subject_list': output.subject_list
        }
        Export_Data_To_Sheets(output.subject_list, is_valid, reasons)
        print(response)
        return jsonify(response)

    
    
    
   
#-------------------------------------------------------------------------------------
# I might use this part later please scroll down



# @app.route('/delete_all', methods=['POST'])
# def delete_all_files():
#     folder = request.json.get('folder')
#     if not folder:
#         return jsonify({'error': 'No folder specified'}), 400
    
#     if folder == 'uploads':
#         folder_path = UPLOAD_FOLDER
#     elif folder == 'processed':
#         folder_path = PROCESSED_FOLDER
#     else:
#         return jsonify({'error': 'Invalid folder specified'}), 400

#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             return jsonify({'error': f'Failed to delete {file_path}. Reason: {e}'}), 500
    
#     return jsonify({'message': 'All files deleted successfully'}), 200

# @app.route('/detections', methods=['GET'])
# def get_detections():
#     print(f'Detected items to send: {detected_class}')  # Debug output
#     return jsonify(list(detected_class))

# @app.route('/processed/<filename>')
# def serve_processed_file(filename):
#     return send_from_directory(PROCESSED_FOLDER, filename)
#-------------------------------------------------------------------------------------


def preprocess(image_path):
    save_dir = 'partialPicSave'
    cropped_images = []  # List to store the cropped images

    # Open the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image was successfully opened
    if img is None:
        print(f"Error: Unable to open image at {image_path}")
        return []

    # Get the image dimensions
    height, width, _ = img.shape

    # Calculate the crop margins
    top_crop = int(height * 0.02)
    bottom_crop = height - top_crop
    left_crop = int(width * 0.15)
    right_crop = width - left_crop

    # Crop the image to remove the top and bottom 2%, and left and right 15%
    img_cropped = img[top_crop:bottom_crop, left_crop:right_crop]

    # Update dimensions after cropping
    height, width, _ = img_cropped.shape

    # Define width ratios and calculate crop widths
    width_ratios = [10, 22, 34, 46, 56, 71]
    crop_widths = [int(width * (ratio / 71)) for ratio in width_ratios]

    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Variables for cropping
    previous_width = 0
    crop_height = int(height * (28 / 53))
    df = pd.DataFrame()
    df = SheetData()
    last=df.iloc[-1]
    
    # Crop and save the top sections
    for i, crop_width in enumerate(crop_widths):
        crop_box = img_cropped[0:crop_height, previous_width:crop_width]
        cropped_image_path = os.path.join(save_dir, f"partialPicSave_{i}_{last['count']}.jpg")
        cv2.imwrite(cropped_image_path, crop_box)
        cropped_images.append((cropped_image_path, crop_box))  # Save tuple of path and image to the list
        print(f"Cropped image section {i} saved at {cropped_image_path}")
        
        previous_width = crop_width

    previous_width = 0
    # Crop and save the bottom sections
    for i, crop_width in enumerate(crop_widths):
        crop_box = img_cropped[crop_height:height, previous_width:crop_width]
        cropped_image_path = os.path.join(save_dir, f"partialPicSave_{i+6}_{last['count']}.jpg")
        cv2.imwrite(cropped_image_path, crop_box)
        cropped_images.append((cropped_image_path, crop_box))  # Save tuple of path and image to the list
        print(f"Cropped image section {i+6} saved at {cropped_image_path}")
        
        previous_width = crop_width

    return cropped_images  # Return the list of cropped images




def process_image(image_path): #main pipe line function
    img_list = preprocess(image_path)
    subject_list = []
    output = data(0, 0, 0, 0, 0, 0, 0 , [])
    for i, (filename, element) in enumerate(img_list):
        output = get_selected_choices(i, element, output, filename)
   
    
    return output


SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = r'Adk/Sheet/oph2024-431003-672c55d29331.json'
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
    
    # If there are no data rows, create an empty DataFrame with the columns
    if df.empty:
        df = pd.DataFrame(columns=columns)
    
    return df

def EditData(df, list, is_valid, reasons):
    color_map = {
        "Pink": 3,
        "Blue": 4,
        "Orange": 5,
        "Brown": 6
    }
    Sem3 = []
    Sem4 = []
    Sem5 = []
    Sem6 = []
    sem_dict = {
        3: Sem3,
        4: Sem4,
        5: Sem5,
        6: Sem6
    }
    print(df.columns)
    new_row = {col: 0 for col in df.columns}
    print(new_row)
    print(len(new_row))
    for outer_key, (inner_value1, inner_value2) in list:
        new_row[inner_value1] = color_map[outer_key]
        chosen_list = sem_dict.get(color_map[outer_key], [])
        chosen_list.append(inner_value1)

    # Convert lists to comma-separated strings
    if Sem3 != [] :
        new_row['sem3'] = ','.join(Sem3)
    if Sem4!= [] :
        new_row['sem4'] = ','.join(Sem4)
    if Sem5 != [] :
        new_row['sem5'] = ','.join(Sem5)
    if Sem6 != [] :
        new_row['sem6'] = ','.join(Sem6)
    
    # Debugging information
    print("New row:", new_row)
    dfl = pd.DataFrame()
    dfl = SheetData()
    last=df.iloc[-1]
    
    df3 = pd.DataFrame([new_row])
    df = df.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)
    #df3.drop([0])
    df3.iloc[0]
    #print("df3:",df3)
    df3.columns = df3.columns.str.strip()
    # Debugging information
    print("Original DataFrame:", df)
    print("New row DataFrame:", df3)
    df3['is_valid'] = is_valid
    if len(reasons) == 0:
        df3['reasons'] = None  # Or any placeholder you prefer
    else:
        df3['reasons'] = [', '.join(reasons)] * len(df3)
    df3['timestamp'] = time.strftime("%Y%m%d-%H%M%S")
    df3['count'] = int(last['count']) + 1
    df3['hostname'] = [hostname]
    df3['system_name'] = [system_name]
    df3['processor'] = [processor]
    df3['version'] = [version]
    df3['machine'] = [machine]
    
    df2 = pd.concat([df, df3])
    
    # Debugging information
    print("Updated DataFrame:", df2)
    
    return df2

def Export_Data_To_Sheets(sbj_ls, is_valid, reasons):
    df = SheetData()
    if df is None:
        print("DataFrame not created. Exiting.")
        return
    df2 = EditData(df, sbj_ls, is_valid, reasons)
    response_date = service.spreadsheets().values().update(
        spreadsheetId=SAMPLE_SPREADSHEET_ID_input,
        valueInputOption='RAW',
        range=SAMPLE_RANGE_NAME,
        body=dict(
            majorDimension='ROWS',
            values=df2.T.reset_index().T.values.tolist())
    ).execute()
    print('Sheet successfully Updated')

#since the real preprocessing isnot  complete so 

# def preprocess(folder_path): # we bypass the preprocessing
#     img_list = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other image formats if needed
#             img_path = os.path.join(folder_path, filename)
#             img = cv2.imread(img_path)
#             if img is not None:
#                 img_list.append((filename, img))
#                 print(filename)
#             else:
#                 print(f"Warning: {filename} could not be read.")
#     return img_list


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


def iou(box1, box2):
    # Calculate the intersection area
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the intersection over union
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def get_selected_choices(i, img,output:data , filename):
    img_rgb = img
    model = YOLO('best6.pt')
    results = model(img_rgb)
    detections = results[0].boxes

    code_boxes = []
    sticker_boxes = []

    img_area = img.shape[0] * img.shape[1]
    max_area = 0.05 * img_area

    for det in detections:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        score = det.conf.item()
        class_id = int(det.cls.item())

        bbox = [x1, y1, x2, y2]
        bbox_area = (x2 - x1) * (y2 - y1)

        if class_id == 4:  # Code
            if bbox_area <= max_area:
                code_boxes.append((bbox, score))
            else:
                print(f"Dropping code box larger than 5% of image area: {bbox}")
        else:
            class_names = ['Pink', 'Brown', 'Blue', 'Orange']
            if class_id in range(len(class_names)):
                sticker_boxes.append((bbox, score, class_names[class_id]))

    # Filter out overlapping stickers
    filtered_sticker_boxes = []
    for ii, (box1, score1, color1) in enumerate(sticker_boxes):
        keep = True
        for j, (box2, score2, color2) in enumerate(sticker_boxes):
            if ii != j :
                if iou(box1, box2) > 0.65:
                    if score1 < score2:
                        keep = False
                        print('drop color')
                        print(color1)
                        break
        if keep:
            filtered_sticker_boxes.append((box1, score1, color1))
    
    sticker_boxes = filtered_sticker_boxes

    for sticker_bbox, _, color in sticker_boxes:
        nearest_box = None
        min_distance = float('inf')

        for code_bbox, _ in code_boxes:
            distance = np.linalg.norm(np.array([
                (sticker_bbox[0] + sticker_bbox[2]) / 2 - (code_bbox[0] + code_bbox[2]) / 2,
                (sticker_bbox[1] + sticker_bbox[3]) / 2 - (code_bbox[1] + code_bbox[3]) / 2
            ]))

            if distance < min_distance:
                min_distance = distance
                nearest_box = code_bbox

        sorted_codes = sorted(code_boxes, key=lambda x: x[0][1])

        if nearest_box in [code[0] for code in sorted_codes]:
            order = [code[0] for code in sorted_codes].index(nearest_box) + 1
            output.subject_list.append([color, subjects[index_table[i] + order - 1]])
            output.total +=1
            
            if color == 'Blue':
                output.blue_count +=1
            elif color == 'Brown':
                output.brown_count +=1
            elif color == 'Orange':
                output.orange_count +=1
            else :
                output.pink_count +=1
            if i<=6:
                output.stem_count+=1
            elif i>9:
                output.foreign_count+=1
    
    print('this is the output')
    print(output.blue_count)
    print(output.brown_count)
    print(output.orange_count)
    print(output.pink_count)
    print(output.foreign_count)
    print(output.total)
    print(output.subject_list)
    print(output.stem_count)
    print(i)
    print('end output')
            
            
            

    
    return output

if __name__ == '__main__':
    app.run()

