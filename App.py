import streamlit as st
import pickle
import os

with open('Tree_model.pkl','rb') as model:
    model_dt = pickle.load(model)

def main():
    st.set_page_config(page_title="Car Purchase Prediction", page_icon="⚕️", layout="centered",initial_sidebar_state="auto")

    html_temp = """ 
    <div style ="background-color:pink;padding:11px">
    <h1 style ="color:black;text-align:center;">Prediction Diabetes</h1> 
        </div><br/>
         <h5 style ="text-align:center;">เว็บแอปพลิเคชันการซื้อรถยนต์แบบปรับเหมาะ</h5>
         กรุณากรอกข้อมูลด้านล่างนี้ ให้ครบถ้วน
        """

    html_sidebar = """
        <div style ="background-color:pink;padding:10px"> 
        <h1 style ="color:black;text-align:center;">สาระสุขภาพ</h1>
                <p style ="color:black;">เนื้อหาข้อมูลด้านล่างนี้  มาจากเว็บไซต์ของ โรงพยาบาลศิริราช ปิยมหาราชการุณย์ ที่ให้ความรู้และให้คำปรึกษาด้านสุขภาพต่างๆ ที่เป็นประโยชน์ที่ดีต่อท่าน</p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetes-2" target="_blank">เบาหวาน รู้ทันป้องกันได้</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetic-diet" target="_blank">เบาหวาน ควรทานอย่างไร</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetes-exercise" target="_blank">ออกกำลังกายพิชิตเบาหวาน</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetes-mellitus" target="_blank">ภาวะแทรกซ้อนจากโรคเบาหวาน</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetic-retinopathy" target="_blank">เบาหวานขึ้นจอตา อันตรายแค่ไหน</a></p>
                <p><a href="https://www.siphhospital.com/th/news/article/share/diabetes-guides" target="_blank">เมื่อเจ็บป่วยควรทำอย่างไร</a></p> 
    </div> 
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.write(html_sidebar, unsafe_allow_html=True)
    Buying = st.selectbox("ช่วงระดับราคาซื้อที่คุณต้องการ", ("vhigh","high","med,","low",))
    if Buying == 'สูงมาก':
        Buying = 1
    elif Buying == 'สูง':
        Buying = 2
    elif Buying == 'ปานกลาง':
        Buying = 3
    elif Buying == 'ต่ำ':
        Buying = 4

    Maint = st.selectbox("ราคาในการซ่อมบำรุงรักษารถ", ("vhigh","high","med,","low"))
    if Maint == 'สูงมาก':
        Maint = 1
    elif Maint == 'สูง':
        Maint = 2
    elif Maint == 'ปานกลาง':
        Maint = 3
    elif Maint == 'ต่ำ':
        Maint = 4

    Doors = st.selectbox("จำนวนประตูที่คุณต้องการ", ("2", "3", "4", "5","มากกว่า5ประตู"))
    if Doors == '2':
        Doors = 1
    elif Doors == '3':
        Doors = 2
    elif Doors == '4':
        Doors = 3
    elif Doors == '5':
        Doors = 4
    elif Doors == 'มากกว่า5ประตู':
        Doors = 5

    Persons = st.selectbox("จำนวนที่นั่งที่คุณต้องการ", ("2","4","มากกว่า4ที่นั่ง"))
    if Persons == '2':
        Persons = 1
    elif Persons == '4':
        Persons = 2
    elif Persons == 'มากกว่า4ที่นั่ง':
        Persons = 3

    Lug = st.selectbox("ขนาดของช่องเก็บของที่คุณต้องการ", ("ขนาดเล็ก","กลาง","ใหญ่"))
    if Lug == 'ขนาดเล็ก':
        Lug = 1
    elif Lug == 'กลาง':
        Lug = 2
    elif Lug == 'ใหญ่':
        Lug = 3

    Safety = st.selectbox("ความปลอดภัยโดยประมาณของรถ", ("ต่ำ","กลาง","สูง"))
    if Safety == 'ต่ำ':
        Safety = 1
    elif Safety == 'กลาง':
        Safety = 2
    elif Safety == 'สูง':
        Safety = 3

    if st.button('ทำนายผล'):
        result = prediction(Buying,Maint,Doors,Persons,Safety)
        if (result == 1):
            st.warning('คุณไม่เหมาะสมกับรถที่เลือก')
        elif (result == 0):
            st.success('ขอแสดงความยินดี คุณเหมาะสมกับรถที่เลือก')

def prediction(Buying,Maint,Doors,Persons, Safety):
    predicted_output = model_dt.predict([[Buying,Maint,Doors,Persons,Safety]])
    return predicted_output

if __name__ == '__main__':
    main()

