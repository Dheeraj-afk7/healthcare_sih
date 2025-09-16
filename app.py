# app_simple.py
import streamlit as st
import pandas as pd
import re
from collections import Counter
import numpy as np

# Simple text preprocessing without NLTK
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    # Simple stop words removal
    stop_words = {'the', 'and', 'or', 'but', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 'an'}
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

def categorize_disease(disease_name):
    disease_name = disease_name.lower()
    categories = {
        'Cancer': ['cancer', 'carcinoma', 'tumor', 'lymphoma', 'leukemia'],
        'Infection': ['infection', 'itis', 'abscess', 'fever', 'pox'],
        'Syndrome': ['syndrome', 'disorder'],
        'Deficiency': ['deficiency', 'anemia'],
        'Poisoning': ['poisoning', 'overdose', 'intoxication'],
        'Injury': ['fracture', 'injury', 'dislocation', 'sprain'],
        'Eye Condition': ['glaucoma', 'cataract', 'vision', 'eye'],
        'Cardiovascular': ['heart', 'cardio', 'hypertension', 'blood pressure'],
        'Endocrine': ['diabetes', 'thyroid', 'hormone'],
        'Pain Condition': ['arthritis', 'pain', 'ache']
    }
    
    for category, keywords in categories.items():
        if any(keyword in disease_name for keyword in keywords):
            return category
    return 'Other'

def calculate_similarity(text1, text2):
    """Simple cosine similarity implementation"""
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

@st.cache_data
def load_data():
    df = pd.read_csv('Diseases_Symptoms.csv')
    df['Cleaned_Symptoms'] = df['Symptoms'].apply(preprocess_text)
    df['Disease_Category'] = df['Name'].apply(categorize_disease)
    return df

def find_similar_diseases(symptoms, df, top_n=5):
    cleaned_symptoms = preprocess_text(symptoms)
    
    # Simple category prediction based on symptom keywords
    symptom_words = set(cleaned_symptoms.split())
    category_keywords = {
        'Infection': {'fever', 'infection', 'swelling', 'redness'},
        'Pain Condition': {'pain', 'ache', 'sore', 'tender'},
        'Cardiovascular': {'chest', 'heart', 'pressure', 'blood'},
        'Respiratory': {'cough', 'breath', 'lung', 'airway'},
        'Gastrointestinal': {'stomach', 'abdominal', 'nausea', 'vomit'}
    }
    
    predicted_category = 'Other'
    max_match = 0
    for category, keywords in category_keywords.items():
        match_count = len(symptom_words.intersection(keywords))
        if match_count > max_match:
            max_match = match_count
            predicted_category = category
    
    # Find similar diseases
    results = []
    for _, row in df.iterrows():
        similarity = calculate_similarity(cleaned_symptoms, row['Cleaned_Symptoms'])
        results.append({
            'disease': row['Name'],
            'symptoms': row['Symptoms'],
            'treatments': row['Treatments'],
            'similarity': similarity,
            'category': row['Disease_Category']
        })
    
    # Sort by similarity and filter by predicted category
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Get top results, prioritizing the predicted category
    top_results = []
    for result in results:
        if result['category'] == predicted_category or len(top_results) < top_n:
            if len(top_results) < top_n * 2:  # Get more results to filter
                top_results.append(result)
    
    # Return top N results
    return predicted_category, top_results[:top_n]

def main():
    st.set_page_config(
        page_title="Disease Prediction System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main-header { font-size: 3rem; color: #2c3e50; text-align: center; }
        .disease-card { background-color: #f8f9fa; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
        .similarity-badge { background-color: #3498db; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading disease database..."):
        df = load_data()
    
    # Sidebar
    st.sidebar.header("Quick Examples")
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
                category, results = find_similar_diseases(symptoms_input, df)
            
            st.success(f"Predicted Category: **{category}**")
            
            for i, disease in enumerate(results, 1):
                with st.expander(f"{i}. {disease['disease']} ({disease['similarity']*100:.1f}% match)"):
                    st.write(f"**Symptoms:** {disease['symptoms']}")
                    st.write(f"**Treatments:** {disease['treatments']}")
                    st.write(f"**Category:** {disease['category']}")
    
    # Statistics
    st.sidebar.header("Database Info")
    st.sidebar.metric("Total Diseases", len(df))
    st.sidebar.metric("Categories", df['Disease_Category'].nunique())
    
    # Disclaimer
    st.markdown("---")
    st.caption("⚠️ **Educational use only.** Always consult healthcare professionals for medical advice.")

if __name__ == "__main__":
    main()