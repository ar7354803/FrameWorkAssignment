# cord19_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import numpy as np

class CORD19Analyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with data path"""
        self.data_path = data_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Load the CORD-19 metadata dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print("✅ Data loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print("❌ File not found. Please check the file path.")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Perform basic data exploration"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic information
        print(f"Dataset dimensions: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Data types
        print("\nData types:")
        print(self.df.dtypes)
        
        # Missing values
        print("\nMissing values in key columns:")
        key_columns = ['title', 'abstract', 'publish_time', 'journal', 'authors']
        missing_data = self.df[key_columns].isnull().sum()
        print(missing_data)
        
        # Basic statistics
        print("\nBasic statistics:")
        print(self.df.describe(include='all'))
        
    def clean_data(self):
        """Clean and prepare the data for analysis"""
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Create a copy for cleaning
        self.cleaned_df = self.df.copy()
        
        # Handle publication dates
        self.cleaned_df['publish_time'] = pd.to_datetime(
            self.cleaned_df['publish_time'], errors='coerce'
        )
        
        # Extract year from publication date
        self.cleaned_df['publication_year'] = self.cleaned_df['publish_time'].dt.year
        
        # Fill missing years with 2020 (most common year for COVID research)
        self.cleaned_df['publication_year'] = self.cleaned_df['publication_year'].fillna(2020)
        
        # Create abstract word count
        self.cleaned_df['abstract_word_count'] = self.cleaned_df['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
        
        # Remove papers with missing titles
        initial_count = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.dropna(subset=['title'])
        final_count = len(self.cleaned_df)
        
        print(f"Removed {initial_count - final_count} papers with missing titles")
        print(f"Final dataset size: {len(self.cleaned_df)} papers")
        
        return self.cleaned_df
    
    def analyze_publications_over_time(self):
        """Analyze publication trends over time"""
        yearly_counts = self.cleaned_df['publication_year'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        yearly_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('COVID-19 Research Publications by Year', fontsize=16, fontweight='bold')
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Papers')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('publications_by_year.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return yearly_counts
    
    def analyze_top_journals(self, top_n=15):
        """Analyze top publishing journals"""
        journal_counts = self.cleaned_df['journal'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 8))
        journal_counts.plot(kind='barh', color='lightcoral')
        plt.title(f'Top {top_n} Journals Publishing COVID-19 Research', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Number of Papers')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return journal_counts
    
    def create_title_wordcloud(self):
        """Create word cloud from paper titles"""
        # Combine all titles
        all_titles = ' '.join(self.cleaned_df['title'].dropna().astype(str))
        
        # Clean the text
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles.lower())
        
        # Remove common stop words
        stop_words = {'study', 'using', 'based', 'analysis', 'research', 
                     'covid', 'coronavirus', 'pandemic', 'model', 'clinical'}
        filtered_words = [word for word in words if word not in stop_words]
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(' '.join(filtered_words))
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words in Paper Titles', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('title_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return most common words
        word_freq = Counter(filtered_words)
        return dict(word_freq.most_common(20))
    
    def analyze_sources(self):
        """Analyze paper distribution by source"""
        source_counts = self.cleaned_df['source_x'].value_counts().head(10)
        
        plt.figure(figsize=(10, 6))
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        plt.title('Paper Distribution by Source', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('source_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return source_counts
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*50)
        print("ANALYSIS REPORT")
        print("="*50)
        
        # Basic statistics
        total_papers = len(self.cleaned_df)
        papers_with_abstracts = self.cleaned_df['abstract'].notna().sum()
        unique_journals = self.cleaned_df['journal'].nunique()
        date_range = f"{int(self.cleaned_df['publication_year'].min())}-{int(self.cleaned_df['publication_year'].max())}"
        
        print(f"Total papers analyzed: {total_papers:,}")
        print(f"Papers with abstracts: {papers_with_abstracts:,} ({papers_with_abstracts/total_papers*100:.1f}%)")
        print(f"Unique journals: {unique_journals}")
        print(f"Publication years: {date_range}")
        
        # Yearly analysis
        yearly = self.analyze_publications_over_time()
        print(f"\nPeak publication year: {yearly.idxmax()} ({yearly.max():,} papers)")
        
        # Journal analysis
        top_journals = self.analyze_top_journals()
        print(f"\nTop journal: {top_journals.index[0]} ({top_journals.iloc[0]:,} papers)")
        
        # Word analysis
        common_words = self.create_title_wordcloud()
        print(f"\nMost common title words: {list(common_words.keys())[:5]}")
        
        # Source analysis
        sources = self.analyze_sources()
        print(f"\nLargest source: {sources.index[0]} ({sources.iloc[0]:,} papers)")

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CORD19Analyzer('data/metadata.csv')
    
    # Run analysis pipeline
    if analyzer.load_data():
        analyzer.explore_data()
        analyzer.clean_data()
        analyzer.generate_report()
