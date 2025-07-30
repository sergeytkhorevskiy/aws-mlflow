#!/usr/bin/env python3
"""
Generate AG News style dataset for text classification.
AG News dataset has 4 classes: World, Sports, Business, Sci/Tech
"""

import pandas as pd
import numpy as np
import random
import os
from typing import List, Tuple

# Sample news headlines and content for each category
WORLD_NEWS = [
    "Global leaders meet to discuss climate change",
    "International trade agreement signed between major economies",
    "UN Security Council addresses regional conflicts",
    "Diplomatic relations established between neighboring countries",
    "World Health Organization issues new guidelines",
    "International summit focuses on economic cooperation",
    "Global pandemic response measures announced",
    "International peace talks begin in conflict zone",
    "World Bank approves funding for development projects",
    "International court ruling on human rights case"
]

SPORTS_NEWS = [
    "Championship team wins dramatic final match",
    "Olympic athlete breaks world record in competition",
    "Professional league announces new season schedule",
    "Team signs star player in major transfer deal",
    "Sports federation introduces new rules for safety",
    "Underdog team advances to championship finals",
    "Athlete receives prestigious sports award",
    "Major sports event draws record attendance",
    "Team coach announces retirement after successful career",
    "Sports organization launches youth development program"
]

BUSINESS_NEWS = [
    "Tech company reports record quarterly profits",
    "Stock market reaches new all-time high",
    "Major merger announced between industry leaders",
    "Startup receives significant venture capital funding",
    "Company launches innovative product line",
    "Economic indicators show strong growth",
    "Business leaders discuss future market trends",
    "Corporate earnings exceed analyst expectations",
    "New business regulations impact industry",
    "Company expands operations to international markets"
]

SCI_TECH_NEWS = [
    "Scientists discover breakthrough in renewable energy",
    "New AI technology revolutionizes industry applications",
    "Research team publishes findings in prestigious journal",
    "Space exploration mission achieves major milestone",
    "Medical breakthrough offers hope for disease treatment",
    "Quantum computing research shows promising results",
    "Biotechnology company develops innovative solution",
    "Climate science study reveals new insights",
    "Robotics technology advances manufacturing processes",
    "Cybersecurity experts develop new protection methods"
]

# Category labels
CATEGORIES = {
    0: "World",
    1: "Sports", 
    2: "Business",
    3: "Sci/Tech"
}

def generate_news_content(category: int, headlines: List[str]) -> Tuple[str, str]:
    """Generate news headline and content for a given category."""
    
    # Select a headline for this category
    if category == 0:
        headline = random.choice(WORLD_NEWS)
        content = f"{headline}. International leaders gathered to address pressing global issues. The meeting focused on strengthening diplomatic ties and promoting international cooperation. Experts believe this development will have significant implications for future international relations."
    elif category == 1:
        headline = random.choice(SPORTS_NEWS)
        content = f"{headline}. The sports world is buzzing with excitement over this remarkable achievement. Fans and analysts alike are praising the exceptional performance. This victory represents a major milestone in the team's history and sets new standards for excellence in the sport."
    elif category == 2:
        headline = random.choice(BUSINESS_NEWS)
        content = f"{headline}. Financial analysts are closely monitoring these developments as they could significantly impact market dynamics. The business community is optimistic about the potential for growth and innovation. This announcement has already influenced investor confidence and market trends."
    elif category == 3:
        headline = random.choice(SCI_TECH_NEWS)
        content = f"{headline}. Researchers and scientists worldwide are excited about the potential applications of this breakthrough. The discovery represents a significant step forward in the field and opens new possibilities for future research. This advancement could revolutionize how we approach related challenges."
    else:
        raise ValueError(f"Invalid category: {category}")
    
    return headline, content

def generate_ag_news_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate AG News style dataset."""
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Generate balanced dataset
    samples_per_category = n_samples // 4
    categories = []
    headlines = []
    contents = []
    
    for category in range(4):
        for _ in range(samples_per_category):
            categories.append(category)
            headline, content = generate_news_content(category, [])
            headlines.append(headline)
            contents.append(content)
    
    # Add some random variation to make it more realistic
    remaining_samples = n_samples - len(categories)
    for _ in range(remaining_samples):
        category = random.randint(0, 3)
        categories.append(category)
        headline, content = generate_news_content(category, [])
        headlines.append(headline)
        contents.append(content)
    
    # Shuffle the data
    indices = list(range(len(categories)))
    random.shuffle(indices)
    
    categories = [categories[i] for i in indices]
    headlines = [headlines[i] for i in indices]
    contents = [contents[i] for i in indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'category': categories,
        'category_name': [CATEGORIES[cat] for cat in categories],
        'headline': headlines,
        'content': contents,
        'text': [f"{h} {c}" for h, c in zip(headlines, contents)]  # Combined text for ML
    })
    
    return df

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add text-based features for ML model."""
    
    # Text length features
    df['headline_length'] = df['headline'].str.len()
    df['content_length'] = df['content'].str.len()
    df['text_length'] = df['text'].str.len()
    
    # Word count features
    df['headline_word_count'] = df['headline'].str.split().str.len()
    df['content_word_count'] = df['content'].str.split().str.len()
    df['text_word_count'] = df['text'].str.split().str.len()
    
    # Average word length
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x else 0
    )
    
    # Capitalization features
    df['capital_ratio'] = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if x else 0
    )
    
    # Punctuation features
    df['punctuation_count'] = df['text'].apply(
        lambda x: sum(1 for c in x if c in '.,!?;:')
    )
    
    return df

def main():
    """Generate and save AG News style dataset."""
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Generate dataset
    print("Generating AG News style dataset...")
    df = generate_ag_news_dataset(n_samples=1000)
    
    # Add text features
    print("Adding text features...")
    df = add_text_features(df)
    
    # Save to CSV
    output_path = 'data/ag_news_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Categories: {list(CATEGORIES.values())}")
    print(f"\nCategory distribution:")
    print(df['category_name'].value_counts())
    
    # Show sample data
    print(f"\nSample data:")
    for i, category in enumerate(CATEGORIES.values()):
        sample = df[df['category_name'] == category].iloc[0]
        print(f"\n{category}:")
        print(f"  Headline: {sample['headline']}")
        print(f"  Content: {sample['content'][:100]}...")
    
    # Show feature statistics
    print(f"\nFeature statistics:")
    numeric_features = ['headline_length', 'content_length', 'text_length', 
                       'headline_word_count', 'content_word_count', 'text_word_count',
                       'avg_word_length', 'capital_ratio', 'punctuation_count']
    print(df[numeric_features].describe())
    
    # Create a simplified version for basic ML models
    print(f"\nCreating simplified version for basic ML...")
    simple_df = df[['category', 'headline_length', 'content_length', 'text_length',
                   'headline_word_count', 'content_word_count', 'text_word_count',
                   'avg_word_length', 'capital_ratio', 'punctuation_count']].copy()
    simple_df.columns = ['target'] + [f'feature_{i}' for i in range(len(simple_df.columns)-1)]
    
    simple_output_path = 'data/dataset.csv'
    simple_df.to_csv(simple_output_path, index=False)
    print(f"Simplified dataset saved to {simple_output_path}")

if __name__ == "__main__":
    main() 