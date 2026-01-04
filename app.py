import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
# --- 1. ตั้งค่าหน้าตาแอป ---
st.markdown("""
    <style>
    /* 1. โหลดฟอนต์ Sarabun */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');

    /* 2. บังคับฟอนต์สารบรรณเฉพาะส่วนที่จำเป็น ไม่ให้รบกวนระบบไอคอน */
    html, body, .stMarkdown, p, label, h1, h2, h3, h4, table, th, td {
        font-family: 'Sarabun', sans-serif !important;
    }

    /* 3. แก้ไขปัญหา Expander ซ้อน (หัวใจสำคัญ) */
    /* ซ่อนข้อความ Accessibility ที่โผล่มาเป็นขยะ เช่น keyboard arrow right */
    div[data-testid="stExpander"] summary span[data-testid="stMarkdownContainer"] p {
        display: none !important;
    }

    /* แสดงเฉพาะข้อความที่เราพิมพ์ลงไปใน expander เท่านั้น */
    div[data-testid="stExpander"] summary > div {
        font-family: 'Sarabun', sans-serif !important;
        font-size: 18px !important;
        font-weight: bold !important;
        color: #2D5A27 !important;
        padding-left: 10px !important;
    }

    /* 4. จัดตำแหน่งลูกศรให้ถูกต้องและไม่ซ้อนทับ */
    div[data-testid="stExpander"] svg[data-testid="stExpanderIcon"] {
        color: #2D5A27 !important;
    }

    /* 5. ตกแต่งปุ่มกดให้ข้อความชัดเจน */
    div.stButton > button {
        font-family: 'Sarabun', sans-serif !important;
        height: 3em !important;
    }
    </style>
    """, unsafe_allow_html=True)
st.set_page_config(page_title="Bio-AI Auto Detector", layout="wide", page_icon="🔍")
st.markdown("""
    <style>
    /* เปลี่ยนสีพื้นหลังหลัก */
    .stApp {
        background-color: #F1F8E9;
    }
    
    /* ปรับแต่ง Font และสีหัวข้อ */
    h1 {
        color: #2D5A27 !important;
        font-family: 'Kanit', sans-serif;
    }
    
    h3 {
        color: #8B5A2B !important;
    }

    /* ปรับแต่งปุ่มกด (Button) */
    div.stButton > button:first-child {
        background-color: #2D5A27;
        color: white;
        border-radius: 10px;
        border: none;
        height: 3em;
        width: 100%;
    }
    
    /* เมื่อเอาเมาส์ไปวางบนปุ่ม */
    div.stButton > button:hover {
        background-color: #8B5A2B;
        color: white;
        border: 1px solid #2D5A27;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* ตกแต่งตารางให้ดูเป็น Dashboard Lab */
    div[data-testid="stTable"] {
        background-color: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #2D5A27;
    }
    
    /* ตกแต่งส่วนหัวตาราง (ถ้าใช้ dataframe) */
    .stDataFrame thead tr th {
        background-color: #2D5A27 !important;
        color: white !important;
        font-family: 'Kanit', sans-serif;
    }

    /* ตกแต่ง Metric (ตัวเลขสรุปด้านล่าง) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #8B5A2B;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    
    /* ตกแต่ง Expander (ส่วนตารางอ้างอิง) */
    .streamlit-expanderHeader {
        background-color: #E8F5E9 !important;
        border-radius: 8px !important;
        font-weight: bold;
        color: #2D5A27;
    }
    </style>
    """, unsafe_allow_html=True)
# --- 2. ฟังก์ชันโหลดโมเดล TFLite ---
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        with open("labels.txt", "r", encoding="utf-8") as f:
            # ดึงเฉพาะชื่อ Class ออกมา (รองรับทั้งแบบมีเลขนำหน้าและไม่มี)
            class_names = [line.strip().split(' ', 1)[-1] for line in f.readlines()]
        return interpreter, input_details, output_details, class_names
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        return None, None, None, None

interpreter, input_details, output_details, class_names = load_tflite_model()

# --- 3. ฐานข้อมูลค่ามาตรฐาน ---
fruit_standards = {
    "กล้วย": {"ph_min": 3.5, "ph_max": 4.0, "ec_min": 6.0},
    "มะละกอ": {"ph_min": 3.2, "ph_max": 3.8, "ec_min": 4.0},
    "ฟักทอง": {"ph_min": 3.4, "ph_max": 4.2, "ec_min": 6.0},
    "สับปะรด": {"ph_min": 3.0, "ph_max": 3.5, "ec_min": 3.5},
    "แตงโม": {"ph_min": 3.4, "ph_max": 4.5, "ec_min": 2.5}
}
# --- 4. ส่วนหน้าจอแอป ---
col_logo, col_text = st.columns([1, 5]) # แบ่งพื้นที่เป็น 2 ส่วน (ส่วนโลโก้กว้าง 1 ส่วน, ชื่อกว้าง 5 ส่วน)

with col_logo:
    try:
        # ใส่ชื่อไฟล์โลโก้ของคุณที่นี่
        logo = Image.open("school_logo.jpg") 
        st.image(logo, width=120) 
    except:
        st.write("📍 [Logo]") # ถ้ายังไม่มีไฟล์รูป จะขึ้นคำนี้แทน

with col_text:
    st.markdown("### โรงเรียนศิลาลาดวิทยา") # ใส่ชื่อโรงเรียนของคุณ
    st.title("ระบบวิเคราะห์น้ำหมักชีวภาพอัจฉริยะ")
    st.write("โครงงานวิทยาศาสตร์และเทคโนโลยี ระดับชั้นมัธยมศึกษาตอนปลาย")

st.divider() # เส้นคั่นเพื่อความสวยงาม

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📸 1. ข้อมูลกายภาพ")
    uploaded_file = st.file_uploader("เลือกรูปภาพน้ำหมัก", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # --- ปรับขนาดภาพให้พอดี (Crop & Resize) ---
        # กำหนดขนาดที่ต้องการ (กว้าง 300, สูง 300 เป็นสี่เหลี่ยมจตุรัส หรือ 300x400 ตามใจชอบ)
        target_size = (300, 300) 
        
        # ImageOps.fit จะทำการตัดขอบภาพ (Crop) ให้พอดีกับสัดส่วนที่ตั้งไว้โดยไม่ทำให้ภาพเบี้ยว
        image_thumbnail = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        
        # แสดงรูปที่ปรับขนาดแล้ว
        st.image(image_thumbnail, caption="ภาพตัวอย่างที่ใช้ประมวลผล", width=300)

with col2:
    st.subheader("🌡️ 2. ข้อมูลทางเคมีและกลิ่น")
    ph_input = st.slider("ค่า pH (ความเป็นกรด-ด่าง)", 0.0, 14.0, 4.0, step=0.01)
    ec_input = st.number_input("ค่า EC (การนำไฟฟ้า mS/cm)", 0.0, 20.0, 1.0, step=0.01)
    odor_score = st.select_slider(
        "👃 ระดับกลิ่น (1:เหม็นเน่า - 5:หอมเปรี้ยวสมบูรณ์)",
        options=[1, 2, 3, 4, 5],
        value=3
    )

# --- 5. ประมวลผลเมื่อกดปุ่ม ---
if st.button("🚀 เริ่มการวิเคราะห์คุณภาพ"):
    if uploaded_file is not None and interpreter is not None:
        with st.spinner('กำลังประมวลผล...'):
            
            # --- 5.1 AI Inference (จำแนกชนิด) ---
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized, dtype=np.float32)
            normalized_img = (img_array / 127.5) - 1
            input_data = np.expand_dims(normalized_img, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            
            idx = np.argmax(prediction)
            detected_fruit = class_names[idx]
            confidence = prediction[idx]

            # --- 5.2 การวิเคราะห์สถานะและเกณฑ์มาตรฐาน ---
            if detected_fruit in fruit_standards:
                std = fruit_standards[detected_fruit]
                
                # ตรวจสอบระยะการหมัก (Phases)
                if ph_input > 5.0:
                    f_status, f_color = "ระยะเริ่มต้น (Initial Phase)", "blue"
                elif 4.0 <= ph_input <= 5.0:
                    f_status, f_color = "ระยะย่อยสลาย (Active Phase)", "orange"
                else:
                    f_status, f_color = "ระยะคงตัว (Stationary Phase)", "green"

                # ตรวจสอบผ่านเกณฑ์หรือไม่ (Logic Flags)
                ph_pass = std['ph_min'] <= ph_input <= std['ph_max']
                ec_pass = ec_input >= std['ec_min']
                odor_pass = odor_score >= 4
                # สมมติว่า Class 0 ใน AI คือภาพที่พร้อม (ให้ปรับตาม labels.txt ของคุณ)
                ai_pass = (idx == 0) 

                # --- 5.3 การแสดงผลลัพธ์ ---
                st.header(f"📍 ผลการตรวจพบ: {detected_fruit}")
                st.markdown(f"**สถานะการหมัก:** :{f_color}[{f_status}]")
                
                # กรณีผ่านเกณฑ์ทั้งหมด
                if ph_pass and ec_pass and odor_pass:
                    st.success(f"✅ **พร้อมใช้งาน:** น้ำหมัก{detected_fruit} มีคุณภาพดีตามมาตรฐาน")
                    st.balloons()
                # กรณีมีบางอย่างไม่ผ่าน
                else:
                    st.warning(f"⚠️ **แจ้งเตือน:** ยังไม่ผ่านเกณฑ์มาตรฐานบางประการ")
                    if not ph_pass: st.info(f"📌 pH {ph_input} ไม่อยู่ในช่วงมาตรฐาน ({std['ph_min']} - {std['ph_max']})")
                    if not ec_pass: st.info(f"📌 EC {ec_input} ต่ำกว่าเกณฑ์ (ควร >= {std['ec_min']})")
                    if not odor_pass: st.info(f"📌 กลิ่นระดับ {odor_score} ยังไม่ถึงระดับพร้อมใช้")

                # Dashboard สรุปตัวเลข
                m1, m2, m3 = st.columns(3)
                m1.metric("ชนิดผลไม้ (AI)", detected_fruit)
                m2.metric("ค่า pH ปัจจุบัน", f"{ph_input:.2f}") # .2f คือทศนิยม 2 ตำแหน่ง
                m3.metric("ค่า EC ปัจจุบัน", f"{ec_input:.2f} mS/cm")
            
            else:
                st.error(f"❓ ไม่พบข้อมูลมาตรฐานสำหรับ '{detected_fruit}'")

    else:
        st.warning("กรุณาอัปโหลดรูปภาพ")

# --- 6. ตารางอ้างอิง ---
with st.expander("ตารางเกณฑ์มาตรฐานเปรียบเทียบ"):
    # 1. สร้างตารางจากฐานข้อมูล
    df_std = pd.DataFrame(fruit_standards).T
    df_std.index.name = "ชนิดน้ำหมัก"
    df_std.columns = ['pH Min', 'pH Max', 'EC Min (mS/cm)']
    
    # 2. แสดงผลเพียงครั้งเดียว (บรรทัดเดียวจบ)
    st.table(df_std.style.format("{:.2f}"))
    # แสดงผลตารางแบบตกแต่ง   
    st.caption("⚠️ อ้างอิงจากเกณฑ์มาตรฐานการผลิตน้ำหมักชีวภาพ (มกอช.) และผลการทดลองทางวิทยาศาสตร์")