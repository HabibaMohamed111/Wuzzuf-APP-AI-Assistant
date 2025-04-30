import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import io
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import plotly.express as px

# Streamlit page setup

@st.cache_data
def load_job_data():
    try:
        if "scraped_file" in st.session_state and st.session_state.scraped_file is not None:
            file = st.session_state.scraped_file
            file_content = file.read()
            file.seek(0)
            if file.name.endswith('.csv'):
                job_data = pd.read_csv(io.BytesIO(file_content))
            elif file.name.endswith('.xlsx'):
                job_data = pd.read_excel(io.BytesIO(file_content))
            else:
                st.error("⚠️ Unsupported file format. Please upload a CSV or XLSX file.")
                return None

            job_data['Skills'] = job_data['Skills'].apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
            return job_data

        else:
            st.info("ℹ️ No file uploaded. Using default demo data from 'demo_data.csv'.")
            demo = pd.read_csv("demo_data.csv")
            demo['Skills'] = demo['Skills'].apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
            return demo
    except Exception as e:
        st.error(f"⚠️ Failed to load data: {e}")
        return None

st.set_page_config(
    page_title="Wuzzuf AI Helper",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main background style (will be overridden in specific pages)
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://i.postimg.cc/gk5FDcZK/Premium-Vector-Digital-technology-concept-background.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.content-container {
    background-color: rgba(255, 255, 255, 0.92);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.92);
    margin-bottom: 30px;
    border: 1px solid #e0e0e0;
}

/* Rest of your original CSS styles remain exactly the same */
.stButton>button {
    background: linear-gradient(90deg, #ff0000, #ff8000, #ffff00, #80ff00, #00ff80, #00ffff, #0080ff, #0000ff, #8000ff, #ff00ff, #ff0080, #ff0000);
    background-size: 600% 100%;
    animation: rainbow 8s linear infinite;
    border: none;
    color: white !important;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.92);
    width: 100%;
    height: 50px;
    border-radius: 8px;
    font-size: 18px;
    margin: 10px 0;
}

.stButton>button:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(444, 66,55,0.25);
}

@keyframes rainbow {
    0% { background-position: 0% 50% }
    100% { background-position: 100% 50% }
}

@keyframes typewriter {
  0% { width: 0%; }
  100% { width: 100%; }
}


.typewriter {
  font-size: 1.1rem;
  font-weight: bold;
  overflow: hidden;
  white-space: nowrap;
  border-right: 3px solid #ff6600;
  animation: typewriter 4s steps(30) 1s 1 normal both;
  color: #FFD700;
  display: inline-block;
}


@keyframes slide {
  0%   { transform: translateX(-100%); opacity: 0;}
  30%  { transform: translateX(0); opacity: 1;}
  70%  { transform: translateX(0); opacity: 1;}
  100% { transform: translateX(100%); opacity: 0;}
}

.moving-header {
  font-size: 2.8rem;
  font-weight: bold;
  text-align: center;
  animation: slide 8s infinite;
  background: linear-gradient(90deg, #1E88E5, #D81B60, #00897B, #FF8F00, #6A1B9A);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}


@keyframes console {
  0% { opacity: 0; transform: translateY(20px); }
  100% { opacity: 1; transform: translateY(0); }
}

.console-text {
  animation: console 1.5s ease-out forwards;
  font-family: 'Courier New', monospace;
}

</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Variable for page navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Function to load job data from session_state

# Main page of the app (unchanged)
def main_page():
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.markdown("""
    <h1 class="moving-header">🚀 Welcome to Wuzzuf AI Job Assistant 🚀</h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://i.postimg.cc/8k44yfP3/Eilik-Cute-Robot-Pets-Toys-with-Abundant-Emotions-Animations-Mini-Games-Your-Perfect-Desk-Touc.jpg", width=200)

    with col2:
        st.markdown(""" 
            <div style="padding: 15px;">
                <h3 style="color: #C0C0C0; margin-bottom: 15px; font-size: 1.5rem;">🤖 Your Smart Wuzzuf Job Analyst</h3>
                <p style="font-size: 18px; line-height: 1.7; margin-bottom: 10px; font-weight: 600;">
                <span class="typewriter">I'm here to help you analyze job data from Wuzzuf and provide smart insights to support your decisions.</span>
                </p>
                <p style="font-size: 16px; line-height: 1.6; color: #E0E0E0; font-weight: 600;">
                <span class="typewriter">I can help you analyze required skills, salaries, leading companies, and much more!</span>
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.markdown("### 📂 Upload Data Files for Analysis")

    with st.expander("Click to upload your files", expanded=True):
        col1, col2 = st.columns(2)

    with col1:
        uploaded_scraped_file = st.file_uploader(
            "Collected Data from Wuzzuf (required)", 
            type=["csv", "xlsx"], 
            key="main_data_uploader"
        )
        if uploaded_scraped_file is not None:
            st.session_state.scraped_file = uploaded_scraped_file
        elif "scraped_file" in st.session_state:
            st.success(f"✅ File already uploaded: {st.session_state.scraped_file.name}")

    with col2:
        uploaded_additional_file = st.file_uploader(
            "Additional Data (optional)", 
            type=["csv", "xlsx"], 
            key="additional_data_uploader"
        )
        if uploaded_additional_file is not None:
            st.session_state.additional_file = uploaded_additional_file
        elif "additional_file" in st.session_state:
            st.success(f"✅ File already uploaded: {st.session_state.additional_file.name}")


    st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.markdown("### Choose What You Want to Do")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Explore Visualizations 📊", key="viz_button"):
            if "scraped_file" in st.session_state and st.session_state.scraped_file is not None:
                st.session_state.page = "visualization"
                st.rerun()
            else:
                st.warning("⚠️ Please upload a job data file to proceed.")
    with col2:
        if st.button("🔍 Explore Suitable Jobs for Your Skills", key="recommend_button"):
            if "scraped_file" in st.session_state and st.session_state.scraped_file is not None:
                st.session_state.page = "recommender"
                st.rerun()
            else:
                st.warning("⚠️ Please upload a job data file to proceed.")

    st.markdown(""" 
        <div style='text-align: center; padding: 20px; margin-top: 30px;'>
             <hr style='border:1px solid #e0e0e0; margin: 20px 0;'>
             <p style='font-size: 20px; font-weight: 800; color: #e0e0e0;'>Wuzzuf AI Job Assistant</p>
             <p style='font-size: 16px; font-weight: 700; color: #e0e0e0;'>Your Smart Job Market Analyst</p>
             <p style='font-size: 14px; font-weight: 600; color: #e0e0e0;'>Made by Data Mafia 🕵🏼‍♀ | © 2025</p>
        </div>
        """, unsafe_allow_html=True)

# Visualization page with custom background
def visualization_page():
    # Override the background for this page only
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://i.postimg.cc/T3DsRrHz/HD-wallpaper-abstract-circuit-computer-detail-electronic-electronics.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Original visualization page code remains exactly the same
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.markdown("## 📊 Interactive Data Visualizations")
    job_data = load_job_data()
    if job_data is None:
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏢 Top Companies", "💼 Top Job Titles", "📍 Top Cities", 
        "🛠️ Top Skills", "📊 Word Cloud", "📈 Top Regions", "🥧 Pie Charts"
    ])

    num_items = st.slider(
        "🔢 Number of items to display:",
        min_value=5, max_value=20, value=10, key="viz_slider"
    )

    with tab1:
        st.header("Top Companies Hiring")
        top_companies = job_data['Company'].value_counts().head(num_items)
        fig1, ax1 = plt.subplots(figsize=(6,4))
        sns.barplot(x=top_companies.values, y=top_companies.index, palette='BuPu', ax=ax1)
        ax1.set_xlabel('Number of Jobs')
        ax1.set_ylabel('Company')
        ax1.set_title(f'Top {num_items} Hiring Companies')
        st.pyplot(fig1)
        with st.expander("View Raw Data"):
            st.dataframe(top_companies.reset_index().rename(columns={'index': 'Company', 0: 'Count'}))
    
    with tab2:
        st.header("Most Popular Job Titles")
        top_titles = job_data['Title'].value_counts().head(num_items)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.barplot(x=top_titles.values, y=top_titles.index, palette='BuPu', ax=ax2)
        ax2.set_xlabel('Number of Jobs')
        ax2.set_ylabel('Job Title')
        ax2.set_title(f'Top {num_items} Job Titles')
        st.pyplot(fig2)
        with st.expander("View Raw Data"):
            st.dataframe(top_titles.reset_index().rename(columns={'index': 'Job Title', 0: 'Count'}))

    with tab3:
        st.header("Most Popular Cities")
        if 'City' not in job_data.columns:
            st.warning("City column not found in the dataset!")
        else:
            cities = job_data['City'].astype(str).replace('nan', '').str.strip()
            cities = cities[cities != '']
            if not cities.empty:
                top_locations = cities.value_counts().head(num_items)
                fig3, ax3 = plt.subplots(figsize=(6,4))
                sns.barplot(x=top_locations.values, y=top_locations.index, palette='viridis', ax=ax3)
                ax3.set_xlabel('Number of Jobs')
                ax3.set_ylabel('City')
                ax3.set_title(f'Top {num_items} Cities')
                st.pyplot(fig3)
                with st.expander("View Raw Data"):
                    st.dataframe(top_locations.reset_index().rename(columns={'index': 'City', 0: 'Count'}))
            else:
                st.warning("No valid city data found after cleaning!")

    with tab4:
        st.header("Most Important Skills")
        if 'Skills' not in job_data.columns:
            st.warning("Skills column not found in the dataset!")
        else:
            skills = job_data['Skills'].astype(str).replace('nan', '').str.split(',').explode()
            skills = skills.str.strip()
            skills = skills[skills != '']
            if not skills.empty:
                top_skills = skills.value_counts().head(num_items)
                fig4, ax4 = plt.subplots(figsize=(6,4))
                sns.barplot(x=top_skills.values, y=top_skills.index, palette='viridis', ax=ax4)
                ax4.set_xlabel('Count')
                ax4.set_ylabel('Skill')
                ax4.set_title(f'Top {num_items} Skills in Demand')
                st.pyplot(fig4)
                with st.expander("View Raw Data"):
                    st.dataframe(top_skills.reset_index().rename(columns={'index': 'Skill', 0: 'Count'}))
            else:
                st.warning("No valid skills found after cleaning!")

    with tab5:
        st.header("Word Cloud of Job Titles")
        if 'Title' not in job_data.columns:
            st.warning("Title column not found in the dataset!")
        else:
            titles = job_data['Title'].astype(str).replace('nan', '')
            if not titles.empty:
                titles_cleaned = titles.str.lower().apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
                text = ' '.join(titles_cleaned)
                wordcloud = WordCloud(
                    width=1000, height=500, background_color='white',
                    stopwords=set(['and', 'with', 'for', 'to', 'in', 'of', 'on', 'a', 'an', 'the']),
                    colormap='magma', max_words=100
                ).generate(text)
                fig5, ax5 = plt.subplots(figsize=(12, 6))
                ax5.imshow(wordcloud, interpolation='bilinear')
                ax5.axis('off')
                ax5.set_title('Most Common Words in Job Titles', fontsize=20)
                st.pyplot(fig5)
            else:
                st.warning("No valid titles found for word cloud!")

    with tab6:
        st.header("Top Regions by Unique Companies")
        if 'Region' not in job_data.columns or 'Company' not in job_data.columns:
            st.warning("Required columns (Region/Company) not found!")
        else:
            regions = job_data[['Region', 'Company']].dropna()
            regions['Region'] = regions['Region'].astype(str).str.strip()
            regions = regions[regions['Region'] != '']
            if not regions.empty:
                top_regions = regions.groupby('Region')['Company'].nunique().sort_values(ascending=False).head(num_items)
                top_regions_df = top_regions.reset_index()
                top_regions_df.columns = ['Region', 'Unique Companies']
                fig6 = px.bar(
                    top_regions_df, x='Region', y='Unique Companies',
                    title=f'Top {num_items} Regions by Unique Companies',
                    color='Unique Companies', text='Unique Companies'
                )
                st.plotly_chart(fig6)
                with st.expander("View Raw Data"):
                    st.dataframe(top_regions_df)
            else:
                st.warning("No valid region data found after cleaning!")

    with tab7:
        st.header("Top Cities Analysis")
        if 'City' not in job_data.columns or 'Company' not in job_data.columns:
            st.warning("Required columns (City/Company) not found!")
        else:
            cities = job_data[['City', 'Company']].dropna()
            cities['City'] = cities['City'].astype(str).str.strip()
            cities = cities[cities['City'] != '']
            if not cities.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Job Postings Distribution")
                    top_cities = cities['City'].value_counts().head(3)
                    if not top_cities.empty:
                        fig7a, ax7a = plt.subplots(figsize=(8,8))
                        ax7a.pie(
                            top_cities, labels=top_cities.index,
                            autopct='%1.1f%%', startangle=90,
                            colors=sns.color_palette('coolwarm', len(top_cities)),
                            explode=[0.1]*len(top_cities), shadow=True
                        )
                        ax7a.set_title('Top 3 Cities by Job Postings', fontsize=14)
                        st.pyplot(fig7a)
                    else:
                        st.warning("No city data available for pie chart")
                with col2:
                    st.subheader("Companies Distribution")
                    company_counts = cities.groupby('City')['Company'].nunique().sort_values(ascending=False).head(3)
                    if not company_counts.empty:
                        fig7b, ax7b = plt.subplots(figsize=(8,8))
                        ax7b.pie(
                            company_counts, labels=company_counts.index,
                            autopct='%1.1f%%', startangle=90,
                            colors=sns.color_palette('coolwarm', len(company_counts)),
                            explode=[0.1]*len(company_counts), shadow=True
                        )
                        ax7b.set_title('Top 3 Cities by Unique Companies', fontsize=14)
                        st.pyplot(fig7b)
                    else:
                        st.warning("No company data available for pie chart")
            else:
                st.warning("No valid city data found for analysis!")

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(""" 
        <div style='text-align: center; padding: 20px; margin-top: 30px;'>
            <hr style='border:1px solid #e0e0e0; margin: 20px 0;'>
            <p style='font-size: 20px; font-weight: 800; color: #e0e0e0;'>Wuzzuf AI Job Assistant</p>
            <p style='font-size: 16px; font-weight: 700; color: #e0e0e0;'>Your Smart Job Market Analyst</p>
            <p style='font-size: 14px; font-weight: 600; color: #e0e0e0;'>Made by Data Mafia 🕵🏼‍♀ | © 2025</p>
        </div>
        """, unsafe_allow_html=True)

# Recommender page with custom background
def recommender_page():
    # Override the background for this page only
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://i.postimg.cc/B6Yp56V5/download-2.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Original recommender page code remains exactly the same
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.markdown("## ✨ Enter your skills to get top job matches")
    job_data = load_job_data()
    if job_data is None:
        return

    user_input = st.text_input("✍️ Enter your skills separated by commas (e.g. Python, SQL, Machine Learning):")

    if user_input:
        if st.button("🚀 Get Recommendations"):
            with st.spinner("Analyzing your skills and matching with jobs..."):
                model = SentenceTransformer('all-MiniLM-L6-v2')
                user_skills = [s.strip() for s in user_input.split(',')]
                user_embedding = model.encode(', '.join(user_skills))

                if 'embedding' not in job_data.columns:
                    st.info("🧠 Calculating embeddings for the job data based on skills...")
                    job_data['embedding'] = job_data['Skills'].apply(lambda x: model.encode(', '.join(x)))

                similarities = cosine_similarity([user_embedding], np.stack(job_data['embedding'])).flatten()
                job_data['similarity'] = similarities
                top_jobs = job_data.sort_values('similarity', ascending=False).head(5)

                st.success("🎯 Top 5 Jobs Matching Your Skills:")
                styled_df = top_jobs[['Title', 'Company', 'Skills', 'similarity']].copy()
                styled_df['Skills'] = styled_df['Skills'].apply(lambda s: ', '.join(s))
                styled_df['similarity'] = styled_df['similarity'].apply(lambda s: f"{s:.2f}")
                st.dataframe(styled_df.reset_index(drop=True), use_container_width=True)

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(""" 
          <div style='text-align: center; padding: 20px; margin-top: 30px;'>
             <hr style='border:1px solid #e0e0e0; margin: 20px 0;'>
             <p style='font-size: 20px; font-weight: 800; color: #e0e0e0;'>Wuzzuf AI Job Assistant</p>
             <p style='font-size: 16px; font-weight: 700; color: #e0e0e0;'>Your Smart Job Market Analyst</p>
             <p style='font-size: 14px; font-weight: 600; color: #e0e0e0;'>Made by Data Mafia 🕵🏼‍♀ | © 2025</p>
          </div>
          """, unsafe_allow_html=True)

# Page navigation control
if st.session_state.page == "home":
    main_page()
elif st.session_state.page == "visualization":
    visualization_page() 
elif st.session_state.page == "recommender":
    recommender_page()
