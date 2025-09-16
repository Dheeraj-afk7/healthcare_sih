# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import math
import os

# Simple text preprocessing without external libraries
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    stop_words = {'the', 'and', 'or', 'but', 'is', 'in', 'it', 'to', 'of', 'for', 
                 'with', 'on', 'at', 'by', 'an', 'a', 'this', 'that', 'are', 'was'}
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

def categorize_disease(disease_name):
    disease_name = disease_name.lower()
    categories = {
        'Cancer': ['cancer', 'carcinoma', 'tumor', 'lymphoma', 'leukemia'],
        'Infection': ['infection', 'itis', 'abscess', 'fever', 'pox', 'flu'],
        'Syndrome': ['syndrome', 'disorder'],
        'Deficiency': ['deficiency', 'anemia'],
        'Poisoning': ['poisoning', 'overdose', 'intoxication'],
        'Injury': ['fracture', 'injury', 'dislocation', 'sprain', 'trauma'],
        'Eye Condition': ['glaucoma', 'cataract', 'vision', 'eye', 'retina'],
        'Cardiovascular': ['heart', 'cardio', 'hypertension', 'blood pressure', 'artery'],
        'Endocrine': ['diabetes', 'thyroid', 'hormone', 'metabolic'],
        'Pain Condition': ['arthritis', 'pain', 'ache', 'neuralgia']
    }
    
    for category, keywords in categories.items():
        if any(keyword in disease_name for keyword in keywords):
            return category
    return 'Other'

def calculate_similarity(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    
    count1 = Counter(words1)
    count2 = Counter(words2)
    
    all_words = set(words1 + words2)
    
    dot_product = 0
    mag1 = 0
    mag2 = 0
    
    for word in all_words:
        dot_product += count1[word] * count2[word]
        mag1 += count1[word] ** 2
        mag2 += count2[word] ** 2
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (math.sqrt(mag1) * math.sqrt(mag2))

@st.cache_data
def load_data():
    """Load data with multiple fallback options"""
    possible_paths = [
        'Diseases_Symptoms.csv',
        './Diseases_Symptoms.csv',
        'data/Diseases_Symptoms.csv',
        '/mount/src/my_model_app/Diseases_Symptoms.csv',
        '/app/Diseases_Symptoms.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.success(f"Loaded data from: {path}")
            return df
    
    # If file not found, create sample data or show error
    st.error("CSV file not found. Please upload your Diseases_Symptoms.csv file.")
    
    # Create sample data structure for demonstration
    sample_data = {
        'Code': [1, 2, 3],
        'Name': ['Sample Disease 1', 'Sample Disease 2', 'Sample Disease 3'],
        'Symptoms': ['fever headache fatigue', 'cough sore throat', 'abdominal pain nausea'],
        'Treatments': ['rest fluids', 'medication', 'diet change']
    }
    return pd.DataFrame(sample_data)

def find_similar_diseases(symptoms, df, top_n=5):
    cleaned_symptoms = preprocess_text(symptoms)
    
    results = []
    for _, row in df.iterrows():
        similarity = calculate_similarity(cleaned_symptoms, preprocess_text(row['Symptoms']))
        results.append({
            'disease': row['Name'],
            'symptoms': row['Symptoms'],
            'treatments': row['Treatments'],
            'similarity': similarity,
            'category': categorize_disease(row['Name'])
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

def main():
    st.set_page_config(
        page_title="Disease Prediction System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main-header { font-size: 3rem; color: #2c3e50; text-align: center; margin-bottom: 2rem; }
        .disease-card { background-color: #f8f9fa; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; border-left: 5px solid #3498db; }
        .similarity-badge { background-color: #3498db; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.9rem; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Diseases_Symptoms.csv", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        # Load data from default location
        with st.spinner("Loading disease database..."):
            df = load_data()
    
    # Display dataset info
    st.sidebar.header("Dataset Info")
    st.sidebar.metric("Total Diseases", len(df))
    if 'Disease_Category' in df.columns:
        st.sidebar.metric("Categories", df['Disease_Category'].nunique())
    
    # Example symptoms
    st.sidebar.header("Example Symptoms")
    examples = [
        "headache fever fatigue",
        "chest pain shortness of breath", 
        "joint pain swelling stiffness",
        "abdominal pain nausea vomiting"
    ]
    
    for example in examples:
        if st.sidebar.button(example):
            st.session_state.symptoms_input = example
    
    # Main interface
    symptoms_input = st.text_area(
        "Enter symptoms (separated by spaces or commas):",
        height=100,
        placeholder="e.g., headache, fever, fatigue...",
        value=st.session_state.get('symptoms_input', '')
    )
    
    if st.button("🔍 Find Similar Diseases", type="primary"):
        if not symptoms_input.strip():
            st.error("Please enter some symptoms")
        else:
            with st.spinner("Searching for similar diseases..."):
                results = find_similar_diseases(symptoms_input, df)
            
            if not results or results[0]['similarity'] < 0.1:
                st.warning("No good matches found. Try different symptoms.")
            else:
                st.success(f"Found {len(results)} potential matches")
                
                for i, disease in enumerate(results, 1):
                    if disease['similarity'] > 0.1:
                        with st.expander(f"{i}. {disease['disease']} ({disease['similarity']*100:.1f}% match)"):
                            st.write(f"**Category:** {disease['category']}")
                            st.write(f"**Symptoms:** {disease['symptoms']}")
                            st.write(f"**Treatments:** {disease['treatments']}")
    
    # Data preview
    with st.expander("View Dataset Preview"):
        st.dataframe(df.head(10))
    
    # Disclaimer
    st.markdown("---")
    st.caption("""
    ⚠️ **Educational use only.** This tool is for demonstration purposes. 
    Always consult healthcare professionals for medical advice and diagnosis.
    """)

if __name__ == "__main__":
    main()