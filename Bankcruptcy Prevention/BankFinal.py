import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))


# creating a function for Prediction

def bankcrupt_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'BANKCRUPT'
    else:
      return 'NON-BANKCRUPT'
  
    
  
def main():
    def welcome(w):
        st.markdown(f'<p style="background-color:#f4c2c2 ;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{w}</p>', unsafe_allow_html=True)
    welcome("WELCOME ALL")

    def stream(s):
        st.markdown(f'<p style="background-color:tomato;padding:10px ;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{s}</p>', unsafe_allow_html=True)
    stream("STREAMLIT BANKCRUPTCY PREVENTION ML APP")
    
    # giving a title
    def title(t):
        st.markdown(f'<p style="text-align:center;font-size:28px">{t}</p>', unsafe_allow_html=True)
    title("BANKCRUPTCY")
    #st.title(BANKCRUPTCY)

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> STREAMLIT BANKCRUPTCY PREVENTION ML APP </h2>
    </div>
    """
    def head(url):
        st.markdown(f'<p style="background-color:#f4c2c2 ;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

    head("(人◕‿◕) 𝔼𝕟𝕥𝕖𝕣 𝕧𝕒𝕝𝕦𝕖𝕤 𝕥𝕠 𝕔𝕙𝕖𝕔𝕜 (•◡•)")
    
    # getting the input data from the user
    
    
    industrial_risk = float(st.number_input("Industrial_risk"))
    management_risk = float(st.number_input("Management_risk"))
    financial_flexibility= float(st.number_input("Financial_flexibility"))
    credibility = float(st.number_input("Credibility"))
    competitiveness = float(st.number_input("Competitiveness"))
    operating_risk = float(st.number_input("Operating_risk"))
    
    
    # code for Prediction
    result = ''
    
    # creating a button for Prediction
    
    if st.button('PREDICT'):
        result = bankcrupt_prediction([industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk])
        
        
    st.success(result)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
def ty(url):
     st.markdown(f'<p style="background-color:	#f2f3f4;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)
ty('T♥H♥A♥N♥K♥ ♥Y♥O♥U')