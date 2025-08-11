import pandas as pd
from googleapiclient.discovery import build
from google.oauth2 import service_account
import os
import pickle

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

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = r'D:\coding\OPH\OPH_project\Adk\Sheet\oph2024-431003-672c55d29331.json'

SAMPLE_SPREADSHEET_ID_input = '1hVjR1LXUF7OuYxyi40DYwskHD97ChhiHkmttSFnBqQc'
SAMPLE_RANGE_NAME = 'A1:GG1000'

def main():
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
    print(df)

main()

# Correct column names
subject_codes = [item[0] for item in subjects]

# Create DataFrame with a single row
df2 = pd.DataFrame([subject_codes], columns=subject_codes)
df2.drop
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

def EditData(df, list):
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
    
    df2 = pd.concat([df, df3])
    
    # Debugging information
    print("Updated DataFrame:", df2)
    
    return df2

def Export_Data_To_Sheets(sbj_ls):
    df = SheetData()
    if df is None:
        print("DataFrame not created. Exiting.")
        return
    df2 = EditData(df, sbj_ls)
    response_date = service.spreadsheets().values().update(
        spreadsheetId=SAMPLE_SPREADSHEET_ID_input,
        valueInputOption='RAW',
        range=SAMPLE_RANGE_NAME,
        body=dict(
            majorDimension='ROWS',
            values=df2.T.reset_index().T.values.tolist())
    ).execute()
    print('Sheet successfully Updated')

sbj_ls = [['Pink', ('ท30206', 'Test')]]
Export_Data_To_Sheets(sbj_ls)