import streamlit as st
import pandas as pd
import random

def exML():
    st.title('🤓Explain-About-Machine-Learning')
    st.write('ใช้งาน sidebar เพื่อใช้งาน Multi-page ไปยังหน้าอื่นๆ')
    st.markdown("""<h1 style='text-align: center;'>อธิบายเกี่ยวกับ Machine learning</h1>""", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "แหล่งข้อมูล&dataset",
        "การเตรียมข้อมูล",
        "Random forest Model(RF)",
        "Gradient Boosting Regressor Model(GBR)"
    ])
    with tab1:
            st.markdown("""<h4 style='text-align: center;text-indent: 2.5em;'>
   แหล่งข้อมูล&dataset
    </h4>""", unsafe_allow_html=True)
            st.markdown("""<h5 style='text-indent: 2.5em;'>
    แหล่งข้อมูลและที่มา
    </h5>""", unsafe_allow_html=True)
            st.markdown("""<p style='text-align: justify;'>
                          ข้อมูลที่นำมาใช้นั้นมีแหล่งมาจาก <a href="https://www.kaggle.com/datasets/hellbuoy/car-price-prediction" target="_blank">Kaggle</a>
                    โดยที่ข้อมูลนั้นเกี่ยวกับรถยนต์ค่ายต่างๆ รวมถึง spec ของรถยนต์เช่น แรงม้า(Horsepower),ประเภทเชื้อเพลิงที่ใช้(fuel type) เป็นต้น<br>
                    เพื่อนำมาใช้ในการ Regression ราคารถนั้นเอง
                    </p>""", unsafe_allow_html=True)
            st.markdown("""<h5 style='text-indent: 2.5em;'>
    Dataset
    </h5>""", unsafe_allow_html=True)
    
            df = pd.read_csv("dataset/CarPrice.csv")
            st.dataframe(df, use_container_width=True)
            st.markdown("""<p style='text-align: justify;'>
                        <li>1.Car_ID	: รหัสเฉพาะของรถ</li><br>	
                        <li>2.Symboling : คะแนนความเสี่ยงในการประกันภัยที่กําหนด ค่า +3 แสดงว่ารถยนต์มีความเสี่ยง -3 ว่ามันน่าจะค่อนข้างปลอดภัย</li><br>	
                        <li>3.carCompany : ค่ายรถ</li><br>			
                        <li>4.fueltype : ประเภทเชื้อเพลิงที่ใช้</li><br>		
                        <li>5.aspiration : การที่เครื่องยนต์รับอากาศเข้าไป</li><br>		
                        <li>6.doornumber : จำนวนของประตู</li><br>	
                        <li>7.carbody : bodyของรถ</li><br>	 		
                        <li>8.drivewheel : ประเภทล้อขับ</li><br>			
                        <li>9.enginelocation : ตำแหน่งเครื่องยนต์</li><br>	 	
                        <li>10.wheelbase : ระยะห่างระหว่างจุดศูนย์กลางของล้อหน้าและจุดศูนย์กลางของล้อหลัง</li><br>				
                        <li>11.carlength : ความยาวรถ</li><br>	 	
                        <li>12.carwidth	: ความกว้างรถ</li><br>		 		
                        <li>13.carheight : ความสูงรถ</li><br>			
                        <li>14.curbweight : น้ำหนักรถเปล่า</li><br>				
                        <li>15.enginetype : ประเภทของเครื่องยนต์</li><br>			
                        <li>16.cylindernumber : กระบอกสูบเครื่องยนต์</li><br>			
                        <li>17.enginesize : ขนาดเครื่อง</li><br>			
                        <li>18.fuelsystem : ระบบเชื้อเพลิง</li><br>				
                        <li>19.boreratio : อัตราส่วนกระบอกสูบ</li><br>			
                        <li>20.stroke : ปริมาตรทั้งหมดที่ลูกสูบทั้งหมดกวาดไปภายในกระบอกสูบขณะเคลื่อนที่</li><br>			
                        <li>21.compressionratio	: อัตราส่วนการอัดของรถ</li><br>			
                        <li>22.horsepower : แรงม้า</li><br>				
                        <li>23.peakrpm : รอบที่สูงที่สุด</li><br>		
                        <li>24.citympg : ระยะทางวิ่งในเมือง</li><br>				
                        <li>25.highwaympg : ระยะทางบนวิ่งทางหลวง</li><br>			
                        <li>26.price(Dependent variable) : ราคารถ</li><br>				
                        </p>""", unsafe_allow_html=True)
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
            code = '''import pandas as pd
df = pd.read_csv('CarPrice.csv')
df.head()'''
            st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
            st.markdown("""<h5 style='text-indent: 2.5em;'>
    2.เลือกเฉพาะฟีเจอร์สำคัญที่ใช้งานใน Streamlit เท่านั้น
แปลง ค่าประเภทข้อความ (categorical) ให้เป็นตัวเลขด้วย LabelEncoder
แสดงผลลัพธ์หลัง preprocessing
    </h5>""", unsafe_allow_html=True)
            code ='''# 🧹 STEP 3: Preprocessing for Selected Features
from sklearn.preprocessing import LabelEncoder

# เลือกเฉพาะฟีเจอร์ที่ Streamlit ใช้
selected_features = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody',
                     'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber',
                     'fuelsystem', 'wheelbase', 'curbweight', 'enginesize',
                     'horsepower', 'citympg', 'highwaympg']

# Drop คอลัมน์อื่นๆ
df = df[selected_features + ['price']]

# แปลง Label Encoding ให้ตรงกับ Streamlit
for col in df.select_dtypes(include='object').columns:
df[col] = LabelEncoder().fit_transform(df[col])

df.head()'''
            st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
            st.markdown("""<h5 style='text-indent: 2.5em;'>
    3.แยกชุดข้อมูลเป็น Training (80%) และ Testing (20%)
เพื่อใช้ในการฝึกและวัดผลโมเดล
    </h5>""", unsafe_allow_html=True)
            code = '''from sklearn.model_selection import train_test_split

X = df[selected_features]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'''
            st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
    with tab3:
            st.markdown("""<h4 style='text-align: center;text-indent: 2.5em;'>
    Random Forest Model
    </h4>""", unsafe_allow_html=True)
            st.markdown("""<p style='text-align: justify;'>
                          🏞Random Forest คือ model ที่ นำ Decision Tree หลายๆ tree มา Train ร่วมกัน (ตั้งแต่ 10 ต้น ถึง มากกว่า 1000 ต้น)<br>
                    โดยที่แต่ละ tree จะได้รับ feature และ data เป็น subset ของ feature และ data ทั้งหมด<br>
                    แบบ random ตอนทำ prediction ก็ให้แต่ละ Decision Tree ทำ prediction ของใครของมัน<br>
                    และเลือกผล final prediction จากค่า prediction ที่ได้รับการโหวตมากที่สุด! — technique ดังกล่าวเรียกว่า bagging หรือ boostrapping
            </p>""", unsafe_allow_html=True)
            st.markdown("""<h5 style='text-indent: 2.5em;'>
    การเทรน model : Random forest
    </h5>""", unsafe_allow_html=True)
            code = '''from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# RF Model
rf = RandomForestRegressor(
    n_estimators=100, #สร้างต้นไม้ทั้งหมด 100 ต้นใน forest
    max_depth=10,     #จำกัดความลึกของ tree สูงสุดที่ 10
    min_samples_split=5, #node จะแตกกิ่งได้ ต้องมีอย่างน้อย 5 sample
    min_samples_leaf=4,  # leaf node ต้องมีอย่างน้อย 4 sample
    max_features='sqrt', # ใช้ฟีเจอร์แบบสุ่ม โดยเลือก √(n_features) ในแต่ละ tree (ลด overfitting และเพิ่ม diversity)
    random_state=42 #ทำให้ผลการเทรน reproducible
)
rf.fit(X_train, y_train) #⚙ Training:
y_pred_rf = rf.predict(X_test) #🔮 Prediction:

print("✅ RF RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf))) #📈 Evaluation:
print("✅ RF R2:", r2_score(y_test, y_pred_rf)) #📈 Evaluation:
#RMSE = ความคลาดเคลื่อนเฉลี่ยในหน่วยของ target
#R² Score = วัดว่าโมเดลอธิบายความแปรปรวนได้ดีแค่ไหน (ใกล้ 1 = ดี)''' 
            st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
    with tab4:
            st.markdown("""<h4 style='text-align: center;text-indent: 2.5em;'>
    Gradient Boosting Regressor Model
    </h4>""", unsafe_allow_html=True)
            st.markdown("""<p style='text-align: justify;'>
                          Gradient Boosting Regressor Model คืออะไร?
Gradient Boosting Regressor (GBR) เป็นอัลกอริธึมหนึ่งในกลุ่มของ Boosting Algorithms ที่ใช้สำหรับงาน Regression (การทำนายค่าต่อเนื่อง)<br>
โดยอาศัยหลักการของ Gradient Boosting ซึ่งเป็นเทคนิคที่รวมเอา weak learners (ตัวเรียนรู้อ่อน) หลายตัวมารวมกันเพื่อสร้าง strong learner (ตัวเรียนรู้ที่แข็งแกร่ง)<br>
GBR ถูกใช้อย่างแพร่หลายในงานที่ต้องการทำนายค่าตัวเลข เช่น<br>
-การพยากรณ์ราคาหุ้น<br>
-การทำนายราคาบ้าน<br>
-การคาดการณ์อุณหภูมิ<br>
-การประมาณค่าความต้องการของลูกค้า<br>
            </p>""", unsafe_allow_html=True)
            st.markdown("""<p style='text-align: justify;'>
                    <b>เหตุผลที่ GBR เป็น Machine Learning</b><br>
-เป็น Supervised Learning<br>

GBR ต้องการ ชุดข้อมูลที่มีป้ายกำกับ (Labeled Data) ซึ่งเป็นคุณสมบัติของ Supervised Learning
โมเดลเรียนรู้จากค่าป้อนเข้า (X) และค่าผลลัพธ์ที่แท้จริง (y)
ใช้หลักการของ Decision Tree<br>

-GBR สร้างแบบจำลองโดยใช้ ต้นไม้ตัดสินใจ (Decision Trees) ซึ่งเป็นเทคนิคที่นิยมใน Machine Learning<br>
ใช้ Boosting ซึ่งเป็นเทคนิค ML ยอดนิยม<br>

-Boosting เป็นเทคนิคที่ใช้โมเดลอ่อน (Weak Learners) หลายตัวมาทำงานร่วมกัน เพื่อสร้างโมเดลที่แข็งแกร่งขึ้น (Strong Learner)<br>
ใช้ Optimization Algorithm (Gradient Descent)<br>

-GBR ใช้วิธี Gradient Descent Optimization เพื่อปรับปรุงโมเดลให้แม่นยำขึ้น ซึ่งเป็นแนวคิดพื้นฐานของ ML<br>
ใช้ในงาน Machine Learning จริง<br>

-GBR ใช้กันอย่างแพร่หลายใน การพยากรณ์ (Prediction) และ การวิเคราะห์ข้อมูล (Data Analysis) เช่น<br>
:การทำนายราคาหุ้น<br>
:การพยากรณ์สภาพอากาศ<br>
:การทำนายแนวโน้มธุรกิจ<br>
            </p>""", unsafe_allow_html=True)
            st.markdown("""<h5 style='text-indent: 2.5em;'>
    การเทรน model : Gradient Boosting Regressor
    </h5>""", unsafe_allow_html=True)
            code = '''from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
gbr = GradientBoostingRegressor(
    learning_rate=0.01, #อัตราการเรียนรู้ — ยิ่งเล็ก โมเดลจะค่อยๆ ปรับตัว, ลด overfitting ได้
    n_estimators=300, #จำนวนต้นไม้ (tree) ที่จะใช้ในการ boosting
    max_depth=3, #	ความลึกสูงสุดของแต่ละ tree (ช่วยลด overfitting)
    min_samples_split=5, #ต้องมีอย่างน้อย 5 sample ถึงจะแตก node ได้
    min_samples_leaf=5, #leaf node ต้องมีอย่างน้อย 5 sample
    subsample=0.8, #ใช้ sample แค่ 80% ของข้อมูลฝึกในแต่ละ tree (ช่วยลด overfitting)
    random_state=42 #เพื่อให้ผลลัพธ์ reproducible (ค่าคงที่ในการสุ่ม)
)
gbr.fit(X_train, y_train) #⚙ Training
y_pred_gbr = gbr.predict(X_test) #🔮 Prediction

📈 Evaluation:
print("✅ GBR RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_gbr)))
print("✅ GBR R2:", r2_score(y_test, y_pred_gbr))
#RMSE = ความคลาดเคลื่อนเฉลี่ยในหน่วยของ target
#R² Score = วัดว่าโมเดลอธิบายความแปรปรวนได้ดีแค่ไหน (ใกล้ 1 = ดี)'''
            st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
                  
