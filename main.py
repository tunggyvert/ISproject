import streamlit as st
from ML import ML
from NN import NN
from exML import exML
from exNN import exNN

def main():
    st.set_page_config(page_title="ุุ6604062620077-jakkapat-ISproject", layout="wide")
    
    # สร้าง session state สำหรับจัดการหน้า
    if 'page' not in st.session_state:
        st.session_state.page = "exML"
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("ใช้งาน sidebar เพื่อใช้งาน Multi-page"
                            " ไปยังหน้าอื่นๆ", 
                            ("Explain-ML", 
                             "Explain-NN", 
                             "Machine Learning Model",
                             "Neural Network Model"))
    
    # อัปเดต session state เมื่อเปลี่ยนหน้า
    st.session_state.page = page
    
    # แสดงเนื้อหาของแต่ละหน้า
    if st.session_state.page == "Explain-ML":
        exML()
    if st.session_state.page == "Explain-NN":
        exNN()
    if st.session_state.page == "Machine Learning Model":
        ML()
    if st.session_state.page == "Neural Network Model":
        NN()

if __name__ == "__main__":
    main()

