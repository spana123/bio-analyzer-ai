import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf # ใช้แค่ดึง Interpreter มาทำงาน
import time

# --- 1. ตั้งค่าหน้าตาแอป ---
st.set_page_config(page_title="Bio-AI TFLite Analyzer", layout="wide", page_icon="🧪")

# --- 2. ฟังก์ชันโหลดโมเดล TFLite ---
@st.cache_resource
def load_tflite_model():
    try:
        # โหลดโมเดล .tflite
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        
        # ดึงรายละเอียด Input/Output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # โหลด Label
        with open("labels.txt", "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]
            
        return interpreter, input_details, output_details, class_names
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดไฟล์ .tflite ได้: {e}")
        return None, None, None, None

# เรียกใช้งานฟังก์ชัน
interpreter, input_details, output_details, class_names = load_tflite_model()

# --- 3. ส่วนหัวของแอป ---
st.title("🌱 ระบบวิเคราะห์น้ำหมักอัจฉริยะ (TFLite Version)")
st.write("การประมวลผลที่รวดเร็วและเสถียรยิ่งขึ้นด้วย TensorFlow Lite")
st.divider()

# --- 4. ส่วนนำเข้าข้อมูล ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 ส่วนที่ 1: วิเคราะห์ภาพถ่าย")
    uploaded_file = st.file_uploader("อัปโหลดรูปผิวหน้าน้ำหมัก", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="ภาพที่นำเข้าสู่ระบบ", use_container_width=True)

with col2:
    st.subheader("📊 ส่วนที่ 2: ข้อมูลทางเคมี")
    fruit_type = st.selectbox("เลือกชนิดผลไม้", ["กล้วย", "มะละกอ", "ฟักทอง", "สับปะรด", "แตงโม"])
    ph = st.slider("ค่า pH", 0.0, 14.0, 4.5, step=0.1)
    ec = st.number_input("ค่า EC (mS/cm)", 0.0, 20.0, 1.0, step=0.1)

# --- 5. การประมวลผล ---
st.divider()

if st.button("🚀 เริ่มการวิเคราะห์ Hybrid"):
    if uploaded_file is not None and interpreter is not None:
        with st.spinner('AI (TFLite) กำลังวิเคราะห์...'):
            # --- 5.1 เตรียมรูปภาพให้ตรงกับที่โมเดลต้องการ (224x224) ---
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized, dtype=np.float32)
            # Normalize (เหมือนตอนเทรนใน Teachable Machine)
            normalized_img = (img_array / 127.5) - 1
            input_data = np.expand_dims(normalized_img, axis=0)

            # --- 5.2 รันโมเดล TFLite ---
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            
            index = np.argmax(prediction)
            label = class_names[index]
            confidence = prediction[index]

            # --- 5.3 ตรรกะตัดสินใจ ---
            ai_ready = (index == 0) # ปรับตามตำแหน่ง Ready ใน labels.txt
            chemical_ready = (3.0 <= ph <= 4.0) and (ec >= 2.0)

            # --- 5.4 แสดงผลลัพธ์ ---
            st.header("📋 ผลสรุปการตรวจสอบ")
            if ai_ready and chemical_ready:
                st.success(f"✅ น้ำหมัก{fruit_type} พร้อมใช้งาน!")
                st.balloons()
            elif ai_ready or chemical_ready:
                st.warning("⚠️ ผลลัพธ์ปานกลาง: ลักษณะกายภาพและเคมียังไม่สอดคล้องกัน")
            else:
                st.error("⏳ ยังไม่พร้อม: กรุณาหมักต่อและปรับสภาพน้ำหมัก")

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("AI วิเคราะห์ว่า", label, f"{confidence:.1%}")
            res_col2.metric("ค่า pH", ph)
            res_col3.metric("ค่า EC", f"{ec} mS/cm")
    else:
        st.warning("กรุณาเตรียมไฟล์ภาพและไฟล์ .tflite ให้พร้อม")

# --- 6. กราฟมาตรฐาน ---
with st.expander("📊 ข้อมูลมาตรฐานอ้างอิง"):
    ref_data = pd.DataFrame({
        'ผลไม้': ['กล้วย', 'มะละกอ', 'ฟักทอง', 'สับปะรด', 'แตงโม'],
        'ค่า EC มาตรฐาน': [8.2, 4.5, 7.1, 3.9, 2.8]
    })
    st.bar_chart(ref_data.set_index('ผลไม้'))