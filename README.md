
# 🤖 Wuzzuf AI Job Assistant

Wuzzuf AI Job Assistant is an intelligent Streamlit application that analyzes job market data from Wuzzuf (Middle Eastern job platform) to provide actionable insights and personalized job recommendations. The app helps job seekers understand market trends and find jobs matching their skills.

## ✨ Key Features

- 📊 **Interactive Visualizations**: Explore top companies, job titles, skills, and locations  
- 🔍 **Skill-Based Job Matching**: Get personalized job recommendations based on your skills  
- 📈 **Market Trends Analysis**: Identify in-demand skills and hiring patterns  
- 🎨 **Stunning UI**: Animated interfaces with customizable themes  
- 🤖 **AI-Powered**: Uses sentence transformers for semantic skill matching  

## 🛠️ Technologies Used

- Python 3  
- Streamlit (Web Framework)  
- Pandas, NumPy (Data Processing)  
- Sentence Transformers (AI Embeddings)  
- Scikit-learn (Cosine Similarity)  
- Matplotlib, Seaborn, Plotly (Visualization)  
- WordCloud (Text Analysis)  

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- pip package manager  

### Installation

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:  
   ```bash
   streamlit run wuzzufAPP.py
   ```

> 📑 **Notes**: When you run the application, please drag and drop `final_cleaned_data(1).csv` file.

## 📊 Sample Analysis

The app provides:

1. Top companies by job postings  
2. Most demanded skills  
3. Job title word clouds  
4. Location-based hiring trends  
5. Skill similarity matching  

## 📂 Data Requirements

Upload a CSV/Excel file containing Wuzzuf job data with these columns (minimum):

1. `Title` - Job title  
2. `Company` - Company name  
3. `Skills` - Comma-separated skills list  
4. (Optional) `City`, `Region` for location analysis  

## 🧙‍♂️ Team  

**Developed with ❤️ by Data Mafia** 🕵🏻‍♀️  

*AI & Data Science Team | Alex, Egypt*
