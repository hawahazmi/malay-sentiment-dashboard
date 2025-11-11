import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.config import TRAIN_RAW, VIDEOS_CSV, COMMENTS_CSV

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration for high-quality plots (for dissertation)
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

print("SOCIAL MEDIA SENTIMENT ANALYSIS - DATASET EXPLORATION")
print()

# ============================================================================
# SECTION 1: TRAINING DATASET EXPLORATION (Labeled Sentiment Data)
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: TRAINING DATASET EXPLORATION")
print("="*80)

# Load the training dataset
# Replace 'training_data.csv' with your actual filename
try:
    df_train = pd.read_csv(TRAIN_RAW)
    print("\nTraining dataset loaded successfully!")
    print(f"File: training_labelled_sentiment.csv")
except FileNotFoundError:
    print("\nFILE NOT FOUND")

print("\n" + "-"*80)
print("1.1 BASIC DATASET INFORMATION")
print("-"*80)

# Display basic info
print(f"\nDataset Shape: {df_train.shape}")
print(f"Number of samples: {df_train.shape[0]:,}")
print(f"Number of features: {df_train.shape[1]}")
print(f"\nColumn Names: {df_train.columns.tolist()}")
print(f"\nData Types:\n{df_train.dtypes}")

# Display first few rows
print(f"\n{'First 10 Rows of Training Dataset':^80}")
print("-"*80)
print(df_train.head(10).to_string(index=True))

# Display last few rows
print(f"\n{'Last 5 Rows of Training Dataset':^80}")
print("-"*80)
print(df_train.tail(5).to_string(index=True))

# Display random sample
print(f"\n{'Random Sample of 5 Rows':^80}")
print("-"*80)
print(df_train.sample(5, random_state=42).to_string(index=True))

print("\n" + "-"*80)
print("1.2 MISSING VALUES ANALYSIS")
print("-"*80)

missing_data = pd.DataFrame({
    'Column': df_train.columns,
    'Missing_Count': df_train.isnull().sum(),
    'Missing_Percentage': (df_train.isnull().sum() / len(df_train) * 100).round(2)
})
print(f"\n{missing_data.to_string(index=False)}")
print(f"\nTotal missing values: {df_train.isnull().sum().sum()}")

print("\n" + "-"*80)
print("1.3 SENTIMENT DISTRIBUTION ANALYSIS")
print("-"*80)

# Assuming sentiment column exists (adjust column name if different)
sentiment_col = 'sentiment'  # Change to 'label' or your actual column name

if sentiment_col in df_train.columns:
    sentiment_counts = df_train[sentiment_col].value_counts()
    sentiment_pct = df_train[sentiment_col].value_counts(normalize=True) * 100

    sentiment_summary = pd.DataFrame({
        'Sentiment': sentiment_counts.index,
        'Count': sentiment_counts.values,
        'Percentage': sentiment_pct.values.round(2)
    })

    print(f"\n{sentiment_summary.to_string(index=False)}")

    # Calculate class imbalance ratio
    max_class = sentiment_counts.max()
    min_class = sentiment_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"\nClass Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")

    if imbalance_ratio > 2.0:
        print("WARNING: Significant class imbalance detected!")
        print("Recommendation: Consider using class weights during training")
    else:
        print("Classes are relatively balanced")

    # Visualization 1: Sentiment Distribution Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # green, red, gray
    bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors[:len(sentiment_counts)])
    ax1.set_xlabel('Sentiment Class', fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontweight='bold')
    ax1.set_title('Training Dataset - Sentiment Distribution', fontweight='bold', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/len(df_train)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    # Pie chart
    explode = [0.05] * len(sentiment_counts)
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors[:len(sentiment_counts)], explode=explode, startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Sentiment Distribution (Percentage)', fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.savefig('1_sentiment_distribution.png', bbox_inches='tight', dpi=300)
    print("\nVisualization saved: 1_sentiment_distribution.png")
    plt.show()

print("\n" + "-"*80)
print("1.4 TEXT LENGTH ANALYSIS")
print("-"*80)

# Assuming text column exists (adjust column name if different)
text_col = 'text'  # Change to 'comment' or your actual column name

if text_col in df_train.columns:
    # Calculate text statistics
    df_train['char_count'] = df_train[text_col].astype(str).apply(len)
    df_train['word_count'] = df_train[text_col].astype(str).apply(lambda x: len(x.split()))

    text_stats = pd.DataFrame({
        'Metric': ['Character Count', 'Word Count'],
        'Mean': [df_train['char_count'].mean(), df_train['word_count'].mean()],
        'Median': [df_train['char_count'].median(), df_train['word_count'].median()],
        'Min': [df_train['char_count'].min(), df_train['word_count'].min()],
        'Max': [df_train['char_count'].max(), df_train['word_count'].max()],
        'Std': [df_train['char_count'].std(), df_train['word_count'].std()]
    })

    print(f"\n{text_stats.round(2).to_string(index=False)}")

    # Text length by sentiment
    if sentiment_col in df_train.columns:
        print(f"\n{'Text Length Statistics by Sentiment':^80}")
        print("-"*80)

        length_by_sentiment = df_train.groupby(sentiment_col).agg({
            'char_count': ['mean', 'median', 'min', 'max'],
            'word_count': ['mean', 'median', 'min', 'max']
        }).round(2)

        print(f"\n{length_by_sentiment.to_string()}")

    # Visualization 2: Text Length Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Character count histogram
    axes[0, 0].hist(df_train['char_count'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df_train['char_count'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {df_train["char_count"].mean():.1f}')
    axes[0, 0].axvline(df_train['char_count'].median(), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {df_train["char_count"].median():.1f}')
    axes[0, 0].set_xlabel('Character Count', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Distribution of Character Count', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Word count histogram
    axes[0, 1].hist(df_train['word_count'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df_train['word_count'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {df_train["word_count"].mean():.1f}')
    axes[0, 1].axvline(df_train['word_count'].median(), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {df_train["word_count"].median():.1f}')
    axes[0, 1].set_xlabel('Word Count', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Distribution of Word Count', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Box plot: Character count by sentiment
    if sentiment_col in df_train.columns:
        sentiment_order = df_train[sentiment_col].value_counts().index.tolist()
        sns.boxplot(data=df_train, x=sentiment_col, y='char_count',
                   order=sentiment_order, ax=axes[1, 0], palette='Set2')
        axes[1, 0].set_xlabel('Sentiment', fontweight='bold')
        axes[1, 0].set_ylabel('Character Count', fontweight='bold')
        axes[1, 0].set_title('Character Count Distribution by Sentiment', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Box plot: Word count by sentiment
        sns.boxplot(data=df_train, x=sentiment_col, y='word_count',
                   order=sentiment_order, ax=axes[1, 1], palette='Set3')
        axes[1, 1].set_xlabel('Sentiment', fontweight='bold')
        axes[1, 1].set_ylabel('Word Count', fontweight='bold')
        axes[1, 1].set_title('Word Count Distribution by Sentiment', fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('2_text_length_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Visualization saved: 2_text_length_analysis.png")
    plt.show()

print("\n" + "-"*80)
print("1.5 DESCRIPTIVE STATISTICS SUMMARY")
print("-"*80)

desc_stats = df_train.describe(include='all').T
print(f"\n{desc_stats.to_string()}")

# ============================================================================
# SECTION 2: YOUTUBE VIDEOS DATASET EXPLORATION
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 2: YOUTUBE VIDEOS DATASET EXPLORATION")
print("="*80)

try:
    df_videos = pd.read_csv(VIDEOS_CSV)
    print("\nYouTube videos dataset loaded successfully!")
    print(f"File: youtube_videos.csv")
except FileNotFoundError:
    print("\nFILE NOT FOUND")

print("\n" + "-"*80)
print("2.1 BASIC DATASET INFORMATION")
print("-"*80)

print(f"\nDataset Shape: {df_videos.shape}")
print(f"Number of videos: {df_videos.shape[0]:,}")
print(f"Number of features: {df_videos.shape[1]}")
print(f"\nColumn Names: {df_videos.columns.tolist()}")
print(f"\nData Types:\n{df_videos.dtypes}")

print(f"\n{'First 5 Videos':^80}")
print("-"*80)
print(df_videos.head().to_string(index=True))

print("\n" + "-"*80)
print("2.2 MISSING VALUES ANALYSIS")
print("-"*80)

missing_videos = pd.DataFrame({
    'Column': df_videos.columns,
    'Missing_Count': df_videos.isnull().sum(),
    'Missing_Percentage': (df_videos.isnull().sum() / len(df_videos) * 100).round(2)
})
print(f"\n{missing_videos.to_string(index=False)}")

print("\n" + "-"*80)
print("2.3 VIDEO ENGAGEMENT STATISTICS")
print("-"*80)

# Engagement metrics analysis
engagement_cols = ['view_count', 'like_count', 'comment_count']
available_engagement = [col for col in engagement_cols if col in df_videos.columns]

if available_engagement:
    engagement_stats = df_videos[available_engagement].describe().T
    print(f"\n{engagement_stats.to_string()}")

    # Calculate engagement rate
    if 'view_count' in df_videos.columns and 'like_count' in df_videos.columns:
        df_videos['engagement_rate'] = (df_videos['like_count'] / df_videos['view_count'] * 100).round(2)
        print(f"\n{'Engagement Rate Statistics':^80}")
        print("-"*80)
        print(f"Mean Engagement Rate: {df_videos['engagement_rate'].mean():.2f}%")
        print(f"Median Engagement Rate: {df_videos['engagement_rate'].median():.2f}%")
        print(f"Max Engagement Rate: {df_videos['engagement_rate'].max():.2f}%")
        print(f"Min Engagement Rate: {df_videos['engagement_rate'].min():.2f}%")

    # Visualization 3: Video Engagement Metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if 'view_count' in df_videos.columns:
        axes[0, 0].hist(df_videos['view_count'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('View Count', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Distribution of Video Views', fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

    if 'like_count' in df_videos.columns:
        axes[0, 1].hist(df_videos['like_count'], bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Like Count', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Distribution of Video Likes', fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

    if 'comment_count' in df_videos.columns:
        axes[1, 0].hist(df_videos['comment_count'], bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Comment Count', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Distribution of Video Comments', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

    if 'engagement_rate' in df_videos.columns:
        axes[1, 1].hist(df_videos['engagement_rate'], bins=30, color='#f39c12', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Engagement Rate (%)', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Distribution of Engagement Rate', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('3_video_engagement_analysis.png', bbox_inches='tight', dpi=300)
    print("\n✓ Visualization saved: 3_video_engagement_analysis.png")
    plt.show()

print("\n" + "-"*80)
print("2.4 TOP PERFORMING VIDEOS")
print("-"*80)

if 'view_count' in df_videos.columns:
    top_videos = df_videos.nlargest(10, 'view_count')
    display_cols = ['title', 'view_count', 'like_count', 'comment_count']
    available_cols = [col for col in display_cols if col in top_videos.columns]
    print(f"\n{'Top 10 Videos by Views':^80}")
    print("-"*80)
    print(top_videos[available_cols].to_string(index=True))

# ============================================================================
# SECTION 3: YOUTUBE COMMENTS DATASET EXPLORATION
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 3: YOUTUBE COMMENTS DATASET EXPLORATION")
print("="*80)

try:
    df_comments = pd.read_csv(COMMENTS_CSV)
    print("\nYouTube comments dataset loaded successfully!")
    print(f"File: youtube_comments.csv")
except FileNotFoundError:
    print("\nFILE NOT FOUND")

print("\n" + "-"*80)
print("3.1 BASIC DATASET INFORMATION")
print("-"*80)

print(f"\nDataset Shape: {df_comments.shape}")
print(f"Number of comments: {df_comments.shape[0]:,}")
print(f"Number of features: {df_comments.shape[1]}")
print(f"\nColumn Names: {df_comments.columns.tolist()}")
print(f"\nData Types:\n{df_comments.dtypes}")

print(f"\n{'First 5 Comments':^80}")
print("-"*80)
print(df_comments.head().to_string(index=True))

print("\n" + "-"*80)
print("3.2 MISSING VALUES ANALYSIS")
print("-"*80)

missing_comments = pd.DataFrame({
    'Column': df_comments.columns,
    'Missing_Count': df_comments.isnull().sum(),
    'Missing_Percentage': (df_comments.isnull().sum() / len(df_comments) * 100).round(2)
})
print(f"\n{missing_comments.to_string(index=False)}")

print("\n" + "-"*80)
print("3.3 COMMENT TEXT LENGTH ANALYSIS")
print("-"*80)

text_col_comments = 'text'  # Adjust if different
if text_col_comments in df_comments.columns:
    df_comments['char_count'] = df_comments[text_col_comments].astype(str).apply(len)
    df_comments['word_count'] = df_comments[text_col_comments].astype(str).apply(lambda x: len(x.split()))

    comment_text_stats = pd.DataFrame({
        'Metric': ['Character Count', 'Word Count'],
        'Mean': [df_comments['char_count'].mean(), df_comments['word_count'].mean()],
        'Median': [df_comments['char_count'].median(), df_comments['word_count'].median()],
        'Min': [df_comments['char_count'].min(), df_comments['word_count'].min()],
        'Max': [df_comments['char_count'].max(), df_comments['word_count'].max()],
        'Std': [df_comments['char_count'].std(), df_comments['word_count'].std()]
    })

    print(f"\n{comment_text_stats.round(2).to_string(index=False)}")

print("\n" + "-"*80)
print("3.4 LANGUAGE DETECTION AND ANALYSIS")
print("-"*80)

# Language detection for comments
try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0  # For reproducibility

    def detect_language(text):
        """Detect language of text with error handling"""
        try:
            if pd.isna(text) or str(text).strip() == '':
                return 'unknown'
            return detect(str(text))
        except LangDetectException:
            return 'unknown'

    print("\nDetecting languages in comments... This may take a moment.")

    # Sample analysis on first 1000 comments (or full dataset if smaller)
    sample_size = min(1000, len(df_comments))
    df_comments_sample = df_comments.head(sample_size).copy()

    print(f"Analyzing {sample_size:,} comments for language detection...")

    df_comments_sample['detected_language'] = df_comments_sample[text_col_comments].apply(detect_language)

    # Language distribution
    lang_counts = df_comments_sample['detected_language'].value_counts()
    lang_pct = (lang_counts / len(df_comments_sample) * 100).round(2)

    # Create language mapping for better readability
    language_names = {
        'ms': 'Malay',
        'en': 'English',
        'id': 'Indonesian',
        'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)',
        'ta': 'Tamil',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'unknown': 'Unknown/Mixed'
    }

    lang_summary = pd.DataFrame({
        'Language_Code': lang_counts.index,
        'Language_Name': [language_names.get(code, code.upper()) for code in lang_counts.index],
        'Count': lang_counts.values,
        'Percentage': lang_pct.values
    })

    print(f"\n{'Language Distribution in Comments':^80}")
    print("-"*80)
    print(f"\n{lang_summary.to_string(index=False)}")

    # Identify Malay vs Non-Malay
    malay_codes = ['ms', 'id']  # Malay and Indonesian (very similar)
    df_comments_sample['is_malay'] = df_comments_sample['detected_language'].isin(malay_codes)

    malay_count = df_comments_sample['is_malay'].sum()
    malay_pct = (malay_count / len(df_comments_sample) * 100)
    non_malay_count = len(df_comments_sample) - malay_count
    non_malay_pct = 100 - malay_pct

    print(f"\n{'Malay vs Non-Malay Comments':^80}")
    print("-"*80)
    print(f"Malay/Indonesian: {malay_count:,} ({malay_pct:.2f}%)")
    print(f"Other Languages: {non_malay_count:,} ({non_malay_pct:.2f}%)")

    # Analyze mixed language (Manglish detection heuristic)
    def detect_manglish(text):
        """Simple heuristic to detect potential Manglish"""
        if pd.isna(text):
            return False
        text = str(text).lower()

        # Common Malay words
        malay_words = ['saya', 'nak', 'tak', 'ada', 'kita', 'dia', 'ini', 'itu',
                       'dengan', 'pada', 'untuk', 'dari', 'yang', 'ke', 'akan',
                       'sudah', 'boleh', 'lah', 'kan', 'pun']

        # Common English words
        english_words = ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                        'can', 'will', 'would', 'should', 'this', 'that', 'very',
                        'good', 'bad', 'like', 'love', 'best', 'nice']

        words = text.split()
        malay_match = sum(1 for word in words if word in malay_words)
        english_match = sum(1 for word in words if word in english_words)

        # If both Malay and English words are present, likely Manglish
        return malay_match > 0 and english_match > 0

    print("\nDetecting code-mixing (Manglish)...")
    df_comments_sample['is_manglish'] = df_comments_sample[text_col_comments].apply(detect_manglish)

    manglish_count = df_comments_sample['is_manglish'].sum()
    manglish_pct = (manglish_count / len(df_comments_sample) * 100)

    print(f"\nPotential Manglish (code-mixed) comments: {manglish_count:,} ({manglish_pct:.2f}%)")

    if manglish_count > 0:
        print("\nNOTE: Code-mixing detected. Preprocessing pipeline should handle Manglish translation.")

    # Sample comments by language
    print(f"\n{'Sample Comments by Language':^80}")
    print("-"*80)

    for lang_code in lang_counts.head(5).index:
        lang_name = language_names.get(lang_code, lang_code.upper())
        sample_comments = df_comments_sample[df_comments_sample['detected_language'] == lang_code][text_col_comments].head(3)

        if len(sample_comments) > 0:
            print(f"\n{lang_name} ({lang_code}):")
            for i, comment in enumerate(sample_comments.values, 1):
                comment_text = str(comment)[:100] + ('...' if len(str(comment)) > 100 else '')
                print(f"  {i}. {comment_text}")

    # Visualization 4.5: Language Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Language distribution bar chart
    top_languages = lang_counts.head(10)
    colors_lang = plt.cm.Set3(range(len(top_languages)))

    bars = axes[0, 0].bar(range(len(top_languages)), top_languages.values, color=colors_lang, edgecolor='black')
    axes[0, 0].set_xticks(range(len(top_languages)))
    axes[0, 0].set_xticklabels([language_names.get(code, code.upper()) for code in top_languages.index],
                                rotation=45, ha='right')
    axes[0, 0].set_ylabel('Number of Comments', fontweight='bold')
    axes[0, 0].set_title('Top 10 Languages Detected in Comments', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}\n({height/len(df_comments_sample)*100:.1f}%)',
                        ha='center', va='bottom', fontsize=8)

    # Pie chart for top languages
    pie_colors = plt.cm.Pastel1(range(len(top_languages)))
    axes[0, 1].pie(top_languages.values,
                   labels=[f"{language_names.get(code, code.upper())}\n{count}"
                          for code, count in zip(top_languages.index, top_languages.values)],
                   autopct='%1.1f%%', colors=pie_colors, startangle=90)
    axes[0, 1].set_title('Language Distribution (Percentage)', fontweight='bold')

    # Malay vs Non-Malay comparison
    malay_data = [malay_count, non_malay_count]
    malay_labels = ['Malay/Indonesian', 'Other Languages']
    malay_colors = ['#2ecc71', '#e74c3c']

    bars2 = axes[1, 0].bar(malay_labels, malay_data, color=malay_colors, edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('Number of Comments', fontweight='bold')
    axes[1, 0].set_title('Malay vs Non-Malay Comments', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}\n({height/len(df_comments_sample)*100:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')

    # Manglish detection results
    manglish_data = [manglish_count, len(df_comments_sample) - manglish_count]
    manglish_labels = ['Manglish\n(Code-Mixed)', 'Single Language']
    manglish_colors = ['#f39c12', '#3498db']

    bars3 = axes[1, 1].bar(manglish_labels, manglish_data, color=manglish_colors, edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('Number of Comments', fontweight='bold')
    axes[1, 1].set_title('Code-Mixing Detection (Manglish)', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}\n({height/len(df_comments_sample)*100:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('4_language_analysis.png', bbox_inches='tight', dpi=300)
    print("\nVisualization saved: 4_language_analysis.png")
    plt.show()

    # Store language analysis results for summary
    language_analysis_results = {
        'total_analyzed': len(df_comments_sample),
        'malay_count': malay_count,
        'malay_pct': malay_pct,
        'non_malay_count': non_malay_count,
        'manglish_count': manglish_count,
        'manglish_pct': manglish_pct,
        'top_language': lang_summary.iloc[0]['Language_Name'] if len(lang_summary) > 0 else 'N/A',
        'unique_languages': len(lang_counts)
    }

except ImportError:
    print("\nlangdetect library not installed.")
    print("Install it using: pip install langdetect")
    print("\nSkipping language detection analysis...")

    language_analysis_results = {
        'total_analyzed': 0,
        'malay_count': 0,
        'malay_pct': 0,
        'non_malay_count': 0,
        'manglish_count': 0,
        'manglish_pct': 0,
        'top_language': 'N/A',
        'unique_languages': 0
    }

print("\n" + "-"*80)
print("3.5 COMMENTS PER VIDEO ANALYSIS")
print("-"*80)

if 'video_id' in df_comments.columns:
    comments_per_video = df_comments['video_id'].value_counts()

    cpv_stats = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std'],
        'Comments per Video': [
            comments_per_video.mean(),
            comments_per_video.median(),
            comments_per_video.min(),
            comments_per_video.max(),
            comments_per_video.std()
        ]
    })

    print(f"\n{cpv_stats.round(2).to_string(index=False)}")

    # Visualization 4: Comments per Video
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(comments_per_video.values, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax1.axvline(comments_per_video.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {comments_per_video.mean():.1f}')
    ax1.set_xlabel('Number of Comments', fontweight='bold')
    ax1.set_ylabel('Number of Videos', fontweight='bold')
    ax1.set_title('Distribution of Comments per Video', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Top 15 videos by comment count
    top_commented = comments_per_video.head(15)
    ax2.barh(range(len(top_commented)), top_commented.values, color='#16a085')
    ax2.set_yticks(range(len(top_commented)))
    ax2.set_yticklabels([f'Video {i+1}' for i in range(len(top_commented))], fontsize=8)
    ax2.set_xlabel('Number of Comments', fontweight='bold')
    ax2.set_title('Top 15 Videos by Comment Count', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig('4_comments_per_video_analysis.png', bbox_inches='tight', dpi=300)
    print("\nVisualization saved: 4_comments_per_video_analysis.png")
    plt.show()

# ============================================================================
# SECTION 4: COMPARATIVE ANALYSIS & SUMMARY
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 4: DATASET SUMMARY & COMPARATIVE ANALYSIS")
print("="*80)

print("\n" + "-"*80)
print("4.1 OVERALL DATASET SUMMARY")
print("-"*80)

summary_table = pd.DataFrame({
    'Dataset': ['Training Data', 'YouTube Videos', 'YouTube Comments'],
    'Total Records': [len(df_train), len(df_videos), len(df_comments)],
    'Total Features': [df_train.shape[1], df_videos.shape[1], df_comments.shape[1]],
    'Missing Values': [
        df_train.isnull().sum().sum(),
        df_videos.isnull().sum().sum(),
        df_comments.isnull().sum().sum()
    ],
    'Memory Usage (MB)': [
        df_train.memory_usage(deep=True).sum() / 1024**2,
        df_videos.memory_usage(deep=True).sum() / 1024**2,
        df_comments.memory_usage(deep=True).sum() / 1024**2
    ]
})

print(f"\n{summary_table.round(2).to_string(index=False)}")

print("\n" + "-"*80)
print("4.2 DATA QUALITY ASSESSMENT")
print("-"*80)

quality_metrics = pd.DataFrame({
    'Dataset': ['Training Data', 'YouTube Videos', 'YouTube Comments'],
    'Completeness (%)': [
        ((1 - df_train.isnull().sum().sum() / (df_train.shape[0] * df_train.shape[1])) * 100),
        ((1 - df_videos.isnull().sum().sum() / (df_videos.shape[0] * df_videos.shape[1])) * 100),
        ((1 - df_comments.isnull().sum().sum() / (df_comments.shape[0] * df_comments.shape[1])) * 100)
    ],
    'Duplicate Rows': [
        df_train.duplicated().sum(),
        df_videos.duplicated().sum(),
        df_comments.duplicated().sum()
    ]
})

print(f"\n{quality_metrics.round(2).to_string(index=False)}")

# Visualization 5: Dataset Overview Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Dataset sizes
datasets = ['Training Data', 'YouTube Videos', 'YouTube Comments']
sizes = [len(df_train), len(df_videos), len(df_comments)]
colors_dataset = ['#3498db', '#e74c3c', '#2ecc71']

axes[0, 0].bar(datasets, sizes, color=colors_dataset, edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('Number of Records', fontweight='bold')
axes[0, 0].set_title('Dataset Sizes Comparison', fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(sizes):
    axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# Missing values comparison
missing_vals = [
    df_train.isnull().sum().sum(),
    df_videos.isnull().sum().sum(),
    df_comments.isnull().sum().sum()
]
axes[0, 1].bar(datasets, missing_vals, color=colors_dataset, edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('Number of Missing Values', fontweight='bold')
axes[0, 1].set_title('Missing Values Comparison', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(missing_vals):
    axes[0, 1].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# Feature count comparison
features = [df_train.shape[1], df_videos.shape[1], df_comments.shape[1]]
axes[1, 0].bar(datasets, features, color=colors_dataset, edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Number of Features', fontweight='bold')
axes[1, 0].set_title('Feature Count Comparison', fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(features):
    axes[1, 0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# Data quality (completeness) comparison
completeness = [
    ((1 - df_train.isnull().sum().sum() / (df_train.shape[0] * df_train.shape[1])) * 100),
    ((1 - df_videos.isnull().sum().sum() / (df_videos.shape[0] * df_videos.shape[1])) * 100),
    ((1 - df_comments.isnull().sum().sum() / (df_comments.shape[0] * df_comments.shape[1])) * 100)
]
axes[1, 1].bar(datasets, completeness, color=colors_dataset, edgecolor='black', alpha=0.7)
axes[1, 1].set_ylabel('Completeness (%)', fontweight='bold')
axes[1, 1].set_title('Data Completeness Comparison', fontweight='bold')
axes[1, 1].set_ylim([0, 105])
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(completeness):
    axes[1, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('5_dataset_comparison_overview.png', bbox_inches='tight', dpi=300)
print("\nVisualization saved: 5_dataset_comparison_overview.png")
plt.show()

# ============================================================================
# SECTION 5: TEMPORAL ANALYSIS (if date columns exist)
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 5: TEMPORAL ANALYSIS")
print("="*80)

# Check for date columns
date_cols_videos = [col for col in df_videos.columns if 'date' in col.lower() or 'published' in col.lower()]
date_cols_comments = [col for col in df_comments.columns if 'date' in col.lower() or 'published' in col.lower()]

if date_cols_videos:
    print("\n" + "-"*80)
    print("5.1 VIDEO PUBLISHING TRENDS")
    print("-"*80)

    date_col = date_cols_videos[0]
    df_videos[date_col] = pd.to_datetime(df_videos[date_col], errors='coerce')

    # Extract temporal features
    df_videos['year'] = df_videos[date_col].dt.year
    df_videos['month'] = df_videos[date_col].dt.month
    df_videos['day_of_week'] = df_videos[date_col].dt.day_name()

    # Videos per year
    videos_per_year = df_videos['year'].value_counts().sort_index()
    print(f"\nVideos Published per Year:")
    print(videos_per_year.to_string())

    # Videos per month
    videos_per_month = df_videos['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Visualization 6: Temporal Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Videos per year
    axes[0, 0].bar(videos_per_year.index, videos_per_year.values,
                   color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Year', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Videos', fontweight='bold')
    axes[0, 0].set_title('Videos Published per Year', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, (year, count) in enumerate(videos_per_year.items()):
        axes[0, 0].text(year, count, str(count), ha='center', va='bottom', fontweight='bold')

    # Videos per month
    month_labels = [month_names[i-1] for i in videos_per_month.index]
    axes[0, 1].bar(month_labels, videos_per_month.values,
                   color='#e74c3c', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Month', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Videos', fontweight='bold')
    axes[0, 1].set_title('Videos Published per Month', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Time series of video publishing
    video_timeline = df_videos.groupby(df_videos[date_col].dt.to_period('M')).size()
    video_timeline.index = video_timeline.index.to_timestamp()
    axes[1, 0].plot(video_timeline.index, video_timeline.values,
                    marker='o', linewidth=2, markersize=6, color='#2ecc71')
    axes[1, 0].set_xlabel('Date', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Videos', fontweight='bold')
    axes[1, 0].set_title('Video Publishing Timeline', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Day of week analysis
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df_videos['day_of_week'].value_counts()
    day_counts = day_counts.reindex(day_order, fill_value=0)

    axes[1, 1].bar(range(len(day_counts)), day_counts.values,
                   color='#f39c12', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xticks(range(len(day_counts)))
    axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[1, 1].set_xlabel('Day of Week', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Videos', fontweight='bold')
    axes[1, 1].set_title('Videos Published by Day of Week', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('6_temporal_analysis.png', bbox_inches='tight', dpi=300)
    print("\nVisualization saved: 6_temporal_analysis.png")
    plt.show()

if date_cols_comments:
    print("\n" + "-"*80)
    print("5.2 COMMENT ACTIVITY TRENDS")
    print("-"*80)

    date_col_comment = date_cols_comments[0]
    df_comments[date_col_comment] = pd.to_datetime(df_comments[date_col_comment], errors='coerce')

    # Comments timeline
    comment_timeline = df_comments.groupby(df_comments[date_col_comment].dt.to_period('M')).size()
    comment_timeline.index = comment_timeline.index.to_timestamp()

    print(f"\nTotal Comments: {len(df_comments):,}")
    print(f"Date Range: {df_comments[date_col_comment].min()} to {df_comments[date_col_comment].max()}")
    print(f"Average Comments per Day: {len(df_comments) / (df_comments[date_col_comment].max() - df_comments[date_col_comment].min()).days:.2f}")

# ============================================================================
# SECTION 6: CORRELATION ANALYSIS
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 6: CORRELATION ANALYSIS")
print("="*80)

print("\n" + "-"*80)
print("6.1 VIDEO METRICS CORRELATION")
print("-"*80)

# Select numeric columns for correlation
numeric_cols_videos = df_videos.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols_videos) > 1:
    correlation_matrix = df_videos[numeric_cols_videos].corr()
    print(f"\n{correlation_matrix.round(3).to_string()}")

    # Visualization 7: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title('Video Metrics Correlation Heatmap', fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('7_correlation_heatmap.png', bbox_inches='tight', dpi=300)
    print("\nVisualization saved: 7_correlation_heatmap.png")
    plt.show()

    # Find strongest correlations
    print("\n" + "-"*80)
    print("6.2 STRONGEST CORRELATIONS")
    print("-"*80)

    # Get upper triangle of correlation matrix
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Find correlations above threshold
    strong_correlations = []
    for column in upper_triangle.columns:
        for index in upper_triangle.index:
            value = upper_triangle.loc[index, column]
            if not pd.isna(value) and abs(value) > 0.3:  # threshold
                strong_correlations.append({
                    'Variable 1': index,
                    'Variable 2': column,
                    'Correlation': value
                })

    if strong_correlations:
        corr_df = pd.DataFrame(strong_correlations)
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        print(f"\n{corr_df.to_string(index=False)}")
    else:
        print("\nNo strong correlations found (threshold: |r| > 0.3)")

# ============================================================================
# SECTION 7: DATA QUALITY RECOMMENDATIONS
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 7: DATA QUALITY ASSESSMENT & RECOMMENDATIONS")
print("="*80)

print("\n" + "-"*80)
print("7.1 DATA QUALITY ISSUES IDENTIFIED")
print("-"*80)

issues = []

# Check for missing values
if df_train.isnull().sum().sum() > 0:
    issues.append(f"Training Data: {df_train.isnull().sum().sum()} missing values detected")

if df_videos.isnull().sum().sum() > 0:
    issues.append(f"YouTube Videos: {df_videos.isnull().sum().sum()} missing values detected")

if df_comments.isnull().sum().sum() > 0:
    issues.append(f"YouTube Comments: {df_comments.isnull().sum().sum()} missing values detected")

# Check for duplicates
if df_train.duplicated().sum() > 0:
    issues.append(f"Training Data: {df_train.duplicated().sum()} duplicate rows found")

if df_videos.duplicated().sum() > 0:
    issues.append(f"YouTube Videos: {df_videos.duplicated().sum()} duplicate rows found")

if df_comments.duplicated().sum() > 0:
    issues.append(f"YouTube Comments: {df_comments.duplicated().sum()} duplicate rows found")

# Check for class imbalance in training data
if sentiment_col in df_train.columns:
    sentiment_counts = df_train[sentiment_col].value_counts()
    imbalance_ratio = sentiment_counts.max() / sentiment_counts.min()
    if imbalance_ratio > 2.0:
        issues.append(f"Training Data: Class imbalance detected (ratio: {imbalance_ratio:.2f})")

# Check for very short or very long texts
if 'char_count' in df_train.columns:
    short_texts = (df_train['char_count'] < 10).sum()
    long_texts = (df_train['char_count'] > 500).sum()
    if short_texts > 0:
        issues.append(f"Training Data: {short_texts} very short texts (< 10 characters)")
    if long_texts > 0:
        issues.append(f"Training Data: {long_texts} very long texts (> 500 characters)")

if issues:
    print("\n⚠ Issues Found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\nNo major data quality issues detected!")

print("\n" + "-"*80)
print("7.2 RECOMMENDATIONS FOR DATA PREPARATION")
print("-"*80)

recommendations = [
    "Handle missing values using appropriate imputation or removal strategies",
    "Remove or investigate duplicate records to ensure data integrity",
    "Address class imbalance using techniques like SMOTE, class weights, or undersampling",
    "Standardize text length by removing very short non-informative comments",
    "Apply comprehensive preprocessing pipeline as outlined in project plan",
    "Consider stratified sampling when splitting data for train/validation/test sets",
    "Validate data quality after each preprocessing step",
    "Create data quality metrics dashboard for ongoing monitoring"
]

print("\n")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

# ============================================================================
# SECTION 8: FINAL SUMMARY REPORT
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 8: EXECUTIVE SUMMARY")
print("="*80)

print(f"""
{'DATASET EXPLORATION SUMMARY REPORT':^80}
{'='*80}

PROJECT: Social Media Sentiment Analysis for Marketing Insights
METHODOLOGY: CRISP-DM Phase 2 - Data Understanding
DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'DATASET OVERVIEW':^80}
{'-'*80}

1. TRAINING DATASET (Labeled Sentiment Data)
   - Total Records: {len(df_train):,}
   - Total Features: {df_train.shape[1]}
   - Missing Values: {df_train.isnull().sum().sum():,}
   - Memory Usage: {df_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   - Sentiment Distribution: {dict(df_train[sentiment_col].value_counts()) if sentiment_col in df_train.columns else 'N/A'}

2. YOUTUBE VIDEOS DATASET
   - Total Videos: {len(df_videos):,}
   - Total Features: {df_videos.shape[1]}
   - Missing Values: {df_videos.isnull().sum().sum():,}
   - Memory Usage: {df_videos.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   - Date Range: {df_videos[date_cols_videos[0]].min() if date_cols_videos else 'N/A'} to {df_videos[date_cols_videos[0]].max() if date_cols_videos else 'N/A'}

3. YOUTUBE COMMENTS DATASET
   - Total Comments: {len(df_comments):,}
   - Total Features: {df_comments.shape[1]}
   - Missing Values: {df_comments.isnull().sum().sum():,}
   - Memory Usage: {df_comments.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   - Average Comment Length: {df_comments['char_count'].mean():.1f} characters

{'KEY FINDINGS':^80}
{'-'*80}

✓ All three datasets have been successfully loaded and explored
✓ Generated {7} high-quality visualizations for dissertation
✓ Identified data quality issues requiring attention in Phase 3
✓ Comprehensive statistical analysis completed
✓ Ready to proceed to Data Preparation phase (CRISP-DM Phase 3)

{'GENERATED VISUALIZATIONS':^80}
{'-'*80}

1. 1_sentiment_distribution.png - Training data sentiment class distribution
2. 2_text_length_analysis.png - Text length statistics and distributions
3. 3_video_engagement_analysis.png - YouTube video engagement metrics
4. 4_comments_per_video_analysis.png - Comment distribution across videos
5. 5_dataset_comparison_overview.png - Comparative analysis of all datasets
6. 6_temporal_analysis.png - Temporal trends in video publishing
7. 7_correlation_heatmap.png - Correlation between video metrics

{'NEXT STEPS':^80}
{'-'*80}

Phase 3: Data Preparation
  → Implement preprocessing pipeline (13 steps as outlined)
  → Handle missing values and duplicates
  → Address class imbalance
  → Prepare data for BiLSTM model training
  → Split data into train/validation/test sets

{'='*80}
END OF REPORT
{'='*80}
""")

print("\n" + "="*80)
print("EXPLORATION COMPLETE!")
print("="*80)
print("\n✓ All visualizations saved successfully")
print("✓ Statistical analysis completed")
print("✓ Ready for dissertation inclusion")
print("\nGenerated Files:")
print("  1. 1_sentiment_distribution.png")
print("  2. 2_text_length_analysis.png")
print("  3. 3_video_engagement_analysis.png")
print("  4. 4_comments_per_video_analysis.png")
print("  5. 5_dataset_comparison_overview.png")
print("  6. 6_temporal_analysis.png")
print("  7. 7_correlation_heatmap.png")
print("\n" + "="*80)