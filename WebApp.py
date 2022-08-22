import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# print "header" in localhost
st.header("Diabetes Detection App")  

# import image
image = Image.open(r"C:\Users\user\Desktop\simple_webapp\diab.jpeg")
st.image(image)

# import dataset
df = pd.read_csv(r"C:\Users\user\Desktop\simple_webapp\diabetes.csv") 

# sub-header
st.subheader("Data")

# show dataframe
st.dataframe(df)

# show description
st.subheader("Data description")
st.write(df.iloc[:, :8].describe().style.background_gradient(cmap = 'copper'))


# ML model
x = df.iloc[:, :8]
y = df.iloc[:, 8]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

model=RandomForestClassifier(n_estimators=500)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

st.subheader("Accuracy of The Trained Model")
st.write(accuracy_score(y_test,y_pred))


#--------------------------------------------------------
# user input

def user_input():
    
    preg=st.slider("Pregnancies",0,20,0)
    glu=st.slider("Glucose",0,200,0)
    bp=st.slider("Blood Pressure",0,130,0)
    sthick=st.slider("Skin Thickness",0,100,0)
    ins=st.slider("Insulin",0.0,1000.0,0.0)
    bmi=st.slider("BMI",0.0,70.0,0.0)
    dpf=st.slider("DPF",0.000,3.000,0.000)
    age=st.slider("Age",0,100,0)
    
    input_dict = {
        "Pregnancies":preg,
        "Glucose":glu,
        "Blood Pressure":bp,
        "Skin Thickness":sthick,
        "Insulin":ins,
        "BMI":bmi,
        "DPF":dpf,
        "Age":age
    }
    
    return pd.DataFrame(input_dict, index = ["User Input values"])

ui = user_input()
st.subheader("Entered user Data.")
st.write(ui)
#-----------------------------------------------------------------------------


# Predictions for User Inputs
st.subheader("Predictions (0 - Non Diabetes, 1 - Diabetes)")
st.write(model.predict(ui))
