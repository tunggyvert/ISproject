import streamlit as st
from PIL import Image

def exNN():
    st.title('🤓Explain-About-Neural Network(Deep Learning)')
    st.write('ใช้งาน sidebar เพื่อใช้งาน Multi-page ไปยังหน้าอื่นๆ')
    st.markdown("""<h1 style='text-align: center;'>อธิบายเกี่ยวกับ Neural Network</h1>""", unsafe_allow_html=True)
    tab1, tab2, tab3= st.tabs([
        "แหล่งข้อมูล&dataset",
        "การเตรียมข้อมูล",
        "การพัฒนาโมเดลด้วย : CNN",
    ])
    with tab1:
        st.markdown("""<h4 style='text-align: center;text-indent: 2.5em;'>
   แหล่งข้อมูล&dataset
    </h4>""", unsafe_allow_html=True)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    แหล่งข้อมูลและที่มา
    </h5>""", unsafe_allow_html=True)
        st.markdown("""<p style='text-align: justify;'>
                          ข้อมูลที่นำมาใช้นั้นมีแหล่งมาจาก <a href="https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification" target="_blank">Kaggle</a>
                    โดยที่ข้อมูลนั้นเกี่ยวกับ Santa และกลุ่มคนที่ไม่ได้เป็น santa <br>
                    เพื่อนำมาใช้ในการทำการ classification(image classification) ว่าเป็น santa หรือไม่เป็น santa นั่นเอง
                    </p>""", unsafe_allow_html=True)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    Dataset
    </h5>""", unsafe_allow_html=True)
        Img = Image.open("imageforweb/ISsanta.PNG")
        resized_img = Img.resize((1496, 475))
        st.image(resized_img)

        st.markdown("""<h5 style='text-align: center;text-indent: 2.5em;'>
    (ตัวอย่าง dataset ที่เอาไว้ใช้เพื่อฝึกว่าเป็น santa)
    </h5>""", unsafe_allow_html=True)
        
        Img = Image.open("imageforweb/ISnotsanta.PNG")
        resized_img = Img.resize((1496, 475))
        st.image(resized_img)

        st.markdown("""<h5 style='text-align: center;text-indent: 2.5em;'>
    (ตัวอย่าง dataset ที่เอาไว้ใช้เพื่อฝึกว่าไม้เป็น santa)
    </h5>""", unsafe_allow_html=True)
        


    with tab2:
        st.markdown("""<h4 style='text-align: center;text-indent: 2.5em;'>
    การเตรียมข้อมูล
    </h4>""", unsafe_allow_html=True)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    How to prep?
    </h5>""", unsafe_allow_html=True)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    1.upload ข้อมูล dataset ที่จะใช้เข้ามาก่อน
        </h5>""", unsafe_allow_html=True)
        code ='''import zipfile
import os
for file in uploaded.keys():
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall('dataset')'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    2.โหลดและเตรียมข้อมูลภาพ (Image Preprocessing)
เตรียมข้อมูลภาพก่อนป้อนเข้าโมเดล ซึ่งเป็นขั้นตอนสำคัญของ Deep Learning ในงาน Image Classification
โมเดล CNN ไม่สามารถรับข้อมูลภาพดิบๆ ได้โดยตรง ต้องปรับแต่งภาพก่อน เช่น resize, normalize, augmentation ฯลฯ
        </h5>""", unsafe_allow_html=True)
        code = '''import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
import os

#ภาพทั้งหมดต้องถูกแยกไว้เป็น 2 โฟลเดอร์นี้ก่อนเริ่มเทรน
train_path = 'dataset/is that santa/train' #โฟลเดอร์ train_path → มีรูปภาพที่ใช้สำหรับ ฝึกโมเดล (Training Data)
test_path = 'dataset/is that santa/test' #โฟลเดอร์ test_path → มีรูปภาพที่ใช้สำหรับ ทดสอบโมเดล (Testing Data)

train_datagen = ImageDataGenerator(rescale=1./255,  #🔹 อธิบาย:rescale=1./255 → Normalize ภาพ (ทำให้ค่าพิกเซลอยู่ในช่วง 0-1 แทนที่จะเป็น 0-255)
                                   rotation_range=20,    #🔹 อธิบาย:rotation_range=20 → หมุนภาพแบบสุ่ม ไม่เกิน 20 องศา (ช่วยให้โมเดลเรียนรู้ดีขึ้น)
                                   zoom_range=0.2,  #🔹อธิบาย: zoom_range=0.2 → ซูมภาพแบบสุ่ม เพิ่มหรือลดขนาดได้ 20%
                                   horizontal_flip=True,    #🔹 อธิบาย: horizontal_flip=True → พลิกภาพด้านซ้าย-ขวาแบบสุ่ม
                                   validation_split=0.2) 

#💡 Data Augmentation คือการเปลี่ยนแปลงภาพแบบสุ่ม เพื่อให้โมเดลเรียนรู้ภาพที่หลากหลายขึ้น
#💡 ช่วย ลด Overfitting เพราะโมเดลจะไม่จดจำแต่ภาพเดิมๆ


train_generator = train_datagen.flow_from_directory(
    train_path,     #🔹 อธิบาย: flow_from_directory() → โหลดภาพจากโฟลเดอร์ และ แปลงเป็นข้อมูลที่โมเดลใช้ได้
    target_size=(150,150),  #🔹 อธิบาย:target_size=(150, 150) → ปรับขนาดภาพเป็น 150x150 พิกเซล
    batch_size=32,  #🔹 อธิบาย:batch_size=32 → ส่งภาพทีละ 32 รูป ให้โมเดลเรียนรู้
    class_mode='categorical',   #🔹 อธิบาย:class_mode='binary' → มี 2 classes (0 = Not Santa / 1 = Santa)
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

#🔹 อธิบาย: สร้างตัวแปลงภาพ (ImageDataGenerator) สำหรับ Testing
#มีแค่ rescale=1./255 เท่านั้น!
#ไม่มี Augmentation (เช่น หมุนภาพ, พลิกภาพ) เพราะข้อมูล Test Set ควรเป็น ของจริง ไม่ถูกเปลี่ยนแปลง
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

    with tab3:
        st.markdown("""<h4 style='text-align: center;text-indent: 2.5em;'>
    การเทรนโมเดล CNN
    </h4>""", unsafe_allow_html=True)
        
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    CNN เป็นยังไง
    </h5>""", unsafe_allow_html=True)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    CNN (Convolutional Neural Network) หรือ โครงข่ายประสาทเทียมแบบคอนโวลูชัน เป็นรูปแบบหนึ่งของโครงข่ายประสาทเทียม (Neural Network) 
    ที่ถูกออกแบบมาเพื่อการประมวลผลข้อมูลที่มีโครงสร้างเชิงพื้นที่ เช่น ภาพและวิดีโอ ได้อย่างมีประสิทธิภาพ<br>
                    
    🔹 โครงสร้างหลักของ CNN : มีโครงสร้างหลักอยู่ 3 ส่วนสำคัญ
                    
        -Convolutional Layer (เลเยอร์คอนโวลูชัน)
        เป็นเลเยอร์ที่ใช้ ฟิลเตอร์ (Kernel) เพื่อสกัดคุณลักษณะ (Feature) สำคัญจากข้อมูล เช่น ขอบ (Edge) หรือรูปแบบ (Pattern) ในภาพ
        ฟิลเตอร์จะเคลื่อนที่ไปทั่วภาพ (Sliding) และคำนวณค่า คอนโวลูชัน (Convolution) เพื่อสร้าง Feature Map
    
        -Pooling Layer (เลเยอร์พูลลิ่ง)
        ใช้เพื่อลดขนาดของข้อมูล ทำให้เครือข่ายมีความซับซ้อนน้อยลงและลดการคำนวณ
        ตัวอย่างวิธีที่นิยมใช้คือ Max Pooling ซึ่งเลือกค่ามากที่สุดในแต่ละบริเวณ
    
        -Fully Connected Layer (เลเยอร์เชื่อมต่อแบบสมบูรณ์)
        เป็นเลเยอร์สุดท้ายที่เชื่อมกับข้อมูลที่ถูกแปลงจากเลเยอร์ก่อนหน้า
        ใช้เพื่อทำการตัดสินใจ เช่น การจำแนกประเภทภาพ
    
        </h5>""", unsafe_allow_html=True)
        
        
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    How to prep?
    </h5>""", unsafe_allow_html=True)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    1.สร้างโมเดล CNN
    </h5>""", unsafe_allow_html=True)
        code = '''model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#✅ ทำอะไร?
#สร้างโมเดล CNN (Convolutional Neural Network) เพื่อแยกแยะภาพ Santa

#Conv2D(32, (3,3)) → ใช้ 32 ตัวกรอง (Filters) ในการจับลักษณะสำคัญของรูปภาพ
#MaxPooling2D(2,2) → ลดขนาดของรูปภาพลง (Downsampling)
#Flatten() → แปลงภาพที่มีหลายมิติให้เป็นเวกเตอร์ 1 มิติ
#Dense(256, activation='relu') → Fully Connected Layer ที่มี 256 นิวรอน
#Dense(1, activation='sigmoid') → ใช้ Sigmoid เพื่อแยก 2 class (Santa / Not Santa)'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    📌2.เทรนโมเดล
    </h5>""", unsafe_allow_html=True)
        code = '''history = model.fit(train_generator, epochs=20, validation_data=test_generator)
#✅ ทำอะไร?

#ใช้ฟังก์ชัน model.fit() เพื่อฝึกโมเดล 20 รอบ (Epochs)
#ใช้ข้อมูลจาก train_generator ในการเรียนรู้
#ใช้ข้อมูลจาก test_generator เพื่อตรวจสอบความแม่นยำ
#history จะเก็บค่าความแม่นยำและ loss เพื่อนำไป plot กราฟได้'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    📌3.แสดงกราฟ Accuracy และ Loss
    </h5>""", unsafe_allow_html=True)
        code = '''plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#✅ ทำอะไร?

#แสดง กราฟความแม่นยำ ของโมเดลขณะเทรน
#เปรียบเทียบระหว่าง Training Accuracy กับ Validation Accuracy
#ถ้าเส้นทั้งสองวิ่งใกล้กัน → โมเดลไม่ Overfitting'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
        st.markdown("""<h5 style='text-indent: 2.5em;'>
    📌4.ประเมินโมเดลบนชุดทดสอบ
    </h5>""", unsafe_allow_html=True)
        code ='''test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

#✅ ทำอะไร?

#นำโมเดลที่เทรนเสร็จแล้วมาทดสอบกับชุดข้อมูล Test Set
#คำนวณค่าความแม่นยำ (test_acc) และค่าเสียหาย (test_loss)
#ใช้เพื่อดูว่าโมเดลสามารถแยกแยะ Santa ได้ดีแค่ไหน
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)




