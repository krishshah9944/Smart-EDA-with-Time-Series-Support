import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

api_key=st.secrets["GROQ_API_KEY"]
st.set_page_config(page_title="EDA Chatbot", page_icon="📊")





st.title("📊 Smart EDA with Time Series Support")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

def detect_time_series(df):
    """Detect time series data by checking for datetime columns with clear formats."""
    time_cols = []
    for col in df.columns:
      
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        
      
        try:
         
            pd.to_datetime(df[col], errors='raise')
            time_cols.append(col)
        except:
            continue
    
    return time_cols if time_cols else None

def generate_eda_code(df, is_time_series=False, time_col=None):
   
    template = """You are a senior data scientist. Perform comprehensive EDA on the provided dataset.
    Generate Python code for complete exploratory data analysis with visualizations.
    
    {time_series_instructions}
    
    Always include these aspects:
    1. Data preview (first 5 rows)
    2. Data quality check (missing values, duplicates)
    3. Basic statistics (describe, info)
    4. Feature distributions (histograms, boxplots)
    5. Correlation analysis (heatmap, pairplot)
    6. Outlier detection
    7. Advanced visualizations
    8. Detailed insights with bullet points
    
    The dataframe is stored in variable 'df'. Use pandas, matplotlib/seaborn, and plotly for visualizations.
    Return ONLY THE CODE wrapped in ```python``` blocks followed by ### INSIGHTS: with markdown-formatted insights.
    
    Dataset sample:
    {sample}
    """

    time_series_section = ""
    if is_time_series and time_col:
        time_series_section = f"""
        SPECIAL TIME SERIES ANALYSIS REQUIRED:
        - Use '{time_col}' as datetime index
        - Resample and analyze trends/seasonality
        - Decompose time series components
        - Calculate rolling statistics
        - Plot autocorrelation and partial autocorrelation
        - Handle missing values in time series context
        - Detect anomalies in temporal patterns
        """
    
    prompt = PromptTemplate(
        input_variables=["sample", "time_series_instructions"],
        template=template,
    )

    # Initialize Groq
    chat = ChatGroq(
        temperature=0.1,
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    chain = LLMChain(llm=chat, prompt=prompt)
    
    sample_data = df.head(3).to_string()
    response = chain.run({
        "sample": sample_data,
        "time_series_instructions": time_series_section
    })
    
    return response


if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
  
    time_cols = detect_time_series(df)
    is_time_series = bool(time_cols)
    selected_time_col = None
    
    if is_time_series:
        st.success(f"⏰ Time series detected in columns: {', '.join(time_cols)}")
        selected_time_col = st.selectbox("Select time series column", time_cols)
        
        
        df[selected_time_col] = pd.to_datetime(df[selected_time_col])
        df = df.sort_values(selected_time_col).set_index(selected_time_col)
    else:
        st.info("🔍 No time series data detected. Proceeding with regular EDA.")
    
    
    with st.expander("Show raw data"):
        st.dataframe(df.head())
    
   
    if st.button("Generate Comprehensive EDA"):
        with st.spinner("Analyzing data..."):
            try:
                response = generate_eda_code(df, is_time_series, selected_time_col)
                
           
                code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
                insights = response.split("### INSIGHTS:")[-1].strip()
                
        
                with st.expander("Generated EDA Code"):
                    for code in code_blocks:
                        st.code(code.strip())
                    
                    
                
                
                with st.expander("Detailed Analysis Insights"):
                    st.markdown(f"""
                    **Comprehensive Insights:**
                    {insights}
                    """)
                
            except Exception as e:
                st.error(f"Error generating EDA: {str(e)}")
else:
    st.warning("Please upload a CSV file to get started")
