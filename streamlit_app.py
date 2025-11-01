# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('data/metadata.csv')
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def clean_data(df):
    """Clean the dataset"""
    df_clean = df.copy()
    
    # Handle dates
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publication_year'] = df_clean['publish_time'].dt.year.fillna(2020)
    
    # Word count
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    # Remove papers without titles
    df_clean = df_clean.dropna(subset=['title'])
    
    return df_clean

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 CORD-19 Research Explorer</h1>', 
                unsafe_allow_html=True)
    st.markdown("Explore COVID-19 research papers from the CORD-19 dataset")
    
    # Load data
    df, error = load_data()
    
    if error:
        st.error(f"Error loading data: {error}")
        st.info("""
        **To run this app:**
        1. Download `metadata.csv` from [Kaggle CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
        2. Create a `data` folder in your project directory
        3. Place `metadata.csv` in the `data` folder
        """)
        return
    
    # Clean data
    df_clean = clean_data(df)
    
    # Sidebar
    st.sidebar.title("Filters & Controls")
    
    # Year range filter
    min_year = int(df_clean['publication_year'].min())
    max_year = int(df_clean['publication_year'].max())
    year_range = st.sidebar.slider(
        "Select publication year range:",
        min_year, max_year, (2020, max_year)
    )
    
    # Journal filter
    top_journals = df_clean['journal'].value_counts().head(20).index.tolist()
    selected_journals = st.sidebar.multiselect(
        "Filter by journal (top 20):",
        options=top_journals,
        default=top_journals[:5]
    )
    
    # Apply filters
    filtered_df = df_clean[
        (df_clean['publication_year'] >= year_range[0]) & 
        (df_clean['publication_year'] <= year_range[1])
    ]
    
    if selected_journals:
        filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", f"{len(filtered_df):,}")
    
    with col2:
        st.metric("Publication Years", f"{year_range[0]} - {year_range[1]}")
    
    with col3:
        papers_with_abstracts = filtered_df['abstract'].notna().sum()
        st.metric("Papers with Abstracts", f"{papers_with_abstracts:,}")
    
    with col4:
        unique_journals = filtered_df['journal'].nunique()
        st.metric("Unique Journals", f"{unique_journals}")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Publication Trends", 
        "📊 Journal Analysis", 
        "🔤 Text Analysis", 
        "📄 Sample Data"
    ])
    
    with tab1:
        st.markdown('<h3 class="section-header">Publication Trends Over Time</h3>', 
                   unsafe_allow_html=True)
        
        yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Publications by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Papers')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
        
        # Monthly trend (if we have the data)
        if 'publish_time' in filtered_df.columns:
            monthly_data = filtered_df.set_index('publish_time').resample('M').size()
            
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            monthly_data.plot(ax=ax2, color='green', linewidth=2)
            ax2.set_title('Monthly Publication Trend')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Papers Published')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig2)
    
    with tab2:
        st.markdown('<h3 class="section-header">Journal Analysis</h3>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top journals
            top_n = st.slider("Number of top journals to show:", 5, 20, 10)
            journal_counts = filtered_df['journal'].value_counts().head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            journal_counts.plot(kind='barh', color='lightcoral', ax=ax)
            ax.set_title(f'Top {top_n} Journals')
            ax.set_xlabel('Number of Papers')
            ax.invert_yaxis()
            st.pyplot(fig)
        
        with col2:
            # Source distribution
            source_counts = filtered_df['source_x'].value_counts().head(8)
            
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
            ax2.set_title('Paper Distribution by Source')
            st.pyplot(fig2)
    
    with tab3:
        st.markdown('<h3 class="section-header">Text Analysis</h3>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Word cloud
            st.subheader("Word Cloud - Paper Titles")
            all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
            
            if all_titles.strip():
                words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles.lower())
                stop_words = {'study', 'using', 'based', 'analysis', 'research', 
                             'covid', 'coronavirus', 'pandemic', 'model', 'clinical'}
                filtered_words = [word for word in words if word not in stop_words]
                
                wordcloud = WordCloud(width=400, height=300, background_color='white',
                                    colormap='viridis', max_words=100).generate(' '.join(filtered_words))
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("No title data available for word cloud")
        
        with col2:
            # Abstract length distribution
            st.subheader("Abstract Length Distribution")
            if filtered_df['abstract_word_count'].sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                filtered_df['abstract_word_count'].hist(bins=30, ax=ax, color='orange', alpha=0.7)
                ax.set_title('Distribution of Abstract Word Counts')
                ax.set_xlabel('Word Count')
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                
                avg_words = filtered_df['abstract_word_count'].mean()
                st.metric("Average Abstract Length", f"{avg_words:.0f} words")
            else:
                st.info("No abstract data available")
    
    with tab4:
        st.markdown('<h3 class="section-header">Sample Research Papers</h3>', 
                   unsafe_allow_html=True)
        
        # Sample data display
        sample_size = st.slider("Number of papers to show:", 5, 50, 10)
        
        display_columns = ['title', 'journal', 'publication_year', 'authors']
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        sample_data = filtered_df[available_columns].head(sample_size)
        st.dataframe(sample_data, use_container_width=True)
        
        # Download option
        csv = filtered_df[available_columns].to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="cord19_filtered_data.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Data Source**: [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) | "
        "**Built with**: Streamlit, Pandas, Matplotlib"
    )

if __name__ == "__main__":
    main()
