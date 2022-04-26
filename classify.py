from re import L
import streamlit as st
import numpy as np
import pandas as pd
from pycaret.classification import load_model, predict_model

def show_input_page():
    st.title("Depression State Classifier")

    st.write("""### Please input the feelings you are experiencing:""")

    gender = st.selectbox("What's your gender?", ('Male', 'Female'))
    age = st.number_input("What's your age?", min_value=0, step=1)
    interest =  st.selectbox("Do you feel little interest or pleasure in doing things?", (0,1,2,3 ))
    depressed =  st.selectbox("Do you feel down, depressed, or hopeless?", (0,1,2,3 ))
    sleep =  st.selectbox("Do you have trouble falling or staying asleep , or sleeping too much?", (0,1,2,3 ))
    energy =  st.selectbox("Do you feel tired or having little energy?", (0,1,2,3 ))
    appetite =  st.selectbox("Do you have poor appetite or overeating?", (0,1,2,3 ))
    yourself =  st.selectbox("Feeling bad about yourself or that you are a failure or have let yourself or your family down?", (0,1,2,3 ))
    concentration =  st.selectbox("Trouble concentrating on things, such as reading the newspaper or watching television?", (0,1,2,3 ))
    notice =  st.selectbox("Moving or speaking so slowly that other people could have noticed. Or being so fighty or restless that you have been moving around a lot more than usual?", (0,1,2,3 ))
    thoughts =  st.selectbox("Thoughts that you would be better off dead, or of hurting yourself?", (0,1,2,3 ))
    problems =  st.selectbox("If you checked off any problems, how difficult have these problems made it for you to do your work, take care of things at home, or get along with other people??", (0,1,2,3 ))
    
        
    submit = st.button("Submit")

    if submit:
        X = [gender, age, interest, depressed, sleep, energy, appetite, yourself, concentration, notice, thoughts, problems]
   
        sample = pd.DataFrame(columns=['Gender', 'Age', 'Little interest or pleasure in doing things',
       'Feeling down, depressed, or hopeless',
       'Trouble falling or staying asleep , or sleeping too much',
       'Feeling tired or having little energy', 'Poor appetite or overeating',
       'Feeling bad about yourself or that you are a failure or have let yourself or your family down',
       'Trouble concentrating on things, such as reading the newspaper or watching television',
       'Moving or speaking so slowly that other people could have noticed. Or being so fighty or restless that you have been moving around a lot more than usual',
       'Thoughts that you would be better off dead, or of hurting yourself',
       'If you checked off any problems, how difficult have these problems made it for you to do your work, take care of things at home, or get along with other people?'], data=[X])
        
        print(sample)
        classifier = load_model('my_best_model')

        testing_sample = predict_model(classifier, data=sample)

        prediction = testing_sample['Label'].to_list()[0]

        st.subheader("Results of Depressive Symptoms: "+prediction+"!")
        



