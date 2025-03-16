import keras
import streamlit as st
import tensorflow as tf
import numpy as np


def NN():
    st.title('🧠Neural Network Model🧠')
    st.write('ใช้งาน sidebar เพื่อใช้งาน Multi-page ไปยังหน้าอื่นๆ')
    st.title("🎅 Santa Image Classification App")
    st.write("อัปโหลดภาพ แล้วระบบจะพยากรณ์ว่าเป็น Santa หรือไม่")
    
    # ✅ โหลดโมเดล
    try:
        model = tf.keras.models.load_model("model/santa_model.h5")
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        return

    # ✅ ชื่อ class
    class_names = ['Not Santa', 'Santa']  # เปลี่ยนได้ตาม class จริง

    # ✅ อัปโหลดภาพ
    uploaded_file = st.file_uploader("📤 อัปโหลดภาพ (jpg/png)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # แสดงภาพ
        st.image(uploaded_file, caption='📷 ภาพที่อัปโหลด', use_container_width=True)

        # โหลดและแปลงภาพ
        img = keras.utils.load_img(uploaded_file, target_size=(150, 150))
        img_array = keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Predict
        prediction = model.predict(img_array)

        # ✅ กรณีโมเดลเป็น Binary (sigmoid)
        if prediction.shape[1] == 1:
            pred_class = 1 if prediction[0][0] > 0.5 else 0
        else:
            pred_class = np.argmax(prediction, axis=1)[0]

        # แสดงผล
        st.success(f"✅ Prediction Result: {class_names[pred_class]}")

