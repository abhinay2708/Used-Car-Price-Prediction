import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Used Car Price Prediction", layout="centered")

def load_models():
    model=joblib.load('car_price_model.joblib')
    return model

try:
    xgb_model=load_models()
except Exception as e:
    st.error(f"Error loading models : {e}")
    st.stop()

st.title("Used Car price Predictor")
st.markdown("Enter teh details below to estimate the car price")

with st.container():
    col1,col2=st.columns(2)
    
    with col1:
        brand=st.selectbox("Brand", ['Ford', 'Hyundai', 'Lexus', 'INFINITI', 'Audi', 'Acura', 'Tesla',
       'Land', 'Toyota', 'Lincoln', 'Jaguar', 'Dodge', 'Nissan',
       'Genesis', 'Chevrolet', 'BMW', 'Mercedes-Benz', 'Kia', 'Jeep',
       'Bentley', 'Honda', 'MINI', 'Porsche', 'Hummer', 'Chrysler',
       'Volvo', 'Cadillac', 'Maserati', 'Volkswagen', 'GMC', 'RAM',
       'Lucid', 'Subaru', 'Alfa', 'Scion', 'Mitsubishi', 'Mazda',
       'Saturn', 'Polestar', 'Buick', 'Aston', 'Lotus', 'Rolls-Royce',
       'Pontiac', 'FIAT', 'Karma', 'Saab', 'Mercury', 'Plymouth', 'smart',
       'Maybach', 'Suzuki'])
        
        fuel_type=st.radio("Fuel Type", ['E85 Flex Fuel', 'Gasoline', 'Hybrid', 'Electric', 'Diesel','Other'])
        
        transmission=st.radio("Transmission",['Automatic', 'Other', 'Manual', 'CVT', 'Single-Speed'])
        
        mileage=st.slider("Mileage",0,202055)
        
        engine_hp=st.slider("Engine HP",70,621)
        
        liters=st.slider("Liters",0.65,7.0)
        
        v_engine=st.selectbox("V_Engine",[1,0])
        
        accident=st.selectbox("Accident",[1,0])
        
        clean_title=st.selectbox("Clean_Title",[1,0])
        
        car_age=st.slider("Car_Age",0,32)
        
        milage_per_year=st.slider("Mileage_per_year",50,49278)
        

if st.button("Predict Price"):
    input_df=pd.DataFrame(
        [
            {
               'brand' : brand,
               'milage' : mileage,
               'fuel_type' : fuel_type,
               'accident' : accident,
               'clean_title' : clean_title,
               'engine_hp' : engine_hp,
               'liters' : liters,
               'v_engine' : v_engine,
               'transmission_type' : transmission,
               'car_age' : car_age,
               'milage_per_year' : milage_per_year
            }
        ]
    )
    
    model= xgb_model
    prediction=model.predict(input_df)[0]
    p=np.exp(prediction)
    
    st.success(f"Estimated Price: ${p:,.2f}")
    st.balloons()