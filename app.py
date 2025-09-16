# app.py
# app_upload.py
import streamlit as st
import pandas as pd
import re
from collections import Counter
import math

# ... [Keep all the functions from above: preprocess_text, categorize_disease, calculate_similarity, find_similar_diseases] ...

def main():
    st.set_page_config(
        page_title="Disease Prediction with Upload",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Disease Prediction System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Diseases_Symptoms.csv file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        
        # Show basic info
        st.sidebar.metric("Diseases in Database", len(df))
        
        # Symptoms input
        symptoms = st.text_input("Enter symptoms:", placeholder="headache, fever, fatigue")
        
        if st.button("Find Similar Diseases") and symptoms:
            results = find_similar_diseases(symptoms, df)
            
            for disease in results:
                st.write(f"**{disease['disease']}** ({disease['similarity']*100:.1f}% match)")
                st.write(f"Symptoms: {disease['symptoms']}")
                st.write(f"Treatments: {disease['treatments']}")
                st.write("---")
                
    else:
        st.info("👆 Please upload a CSV file with columns: Code, Name, Symptoms, Treatments")
        
        # Show sample format
        sample_data = {
            'Code': [1, 2, 3],
            'Name': ['Flu', 'Cold', 'Headache'],
            'Symptoms': ['fever cough fatigue', 'runny nose sneezing', 'head pain'],
            'Treatments': ['rest fluids', 'medication', 'pain relief']
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()