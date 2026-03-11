"""
Sample Dataset Generator
Creates example marketing analytics data for testing
"""

import pandas as pd
import numpy as np


def generate_sample_data(n_rows: int = 500, filename: str = "sample_data.csv"):
    """
    Generate sample marketing analytics data with Simpson's Paradox examples
    
    Args:
        n_rows: Number of rows to generate
        filename: Output filename
    """
    np.random.seed(42)
    
    # Dimensions
    traffic_types = ['Paid', 'Organic', 'Email', 'Social', 'Direct']
    campaigns = ['Campaign_A', 'Campaign_B', 'Campaign_C', 'Campaign_D']
    regions = ['North America', 'Europe', 'Asia', 'South America']
    device_types = ['Desktop', 'Mobile', 'Tablet']
    
    data = {
        'Traffic_Type': np.random.choice(traffic_types, n_rows, 
                                        p=[0.3, 0.25, 0.15, 0.2, 0.1]),
        'Campaign': np.random.choice(campaigns, n_rows),
        'Region': np.random.choice(regions, n_rows),
        'Device': np.random.choice(device_types, n_rows, p=[0.5, 0.4, 0.1])
    }
    
    # Generate correlated metrics
    # Base metrics
    impressions = np.random.exponential(10000, n_rows)
    clicks = impressions * np.random.uniform(0.01, 0.05, n_rows)
    
    # Correlated metrics
    data['Impressions'] = impressions.astype(int)
    data['Clicks'] = clicks.astype(int)
    data['CTR'] = (clicks / impressions * 100)  # Strongly correlated with Clicks
    data['CPC'] = np.random.uniform(0.5, 3.0, n_rows)
    data['Spend'] = data['Clicks'] * data['CPC']  # Correlated with Clicks
    data['Conversions'] = clicks * np.random.uniform(0.05, 0.15, n_rows)
    data['Revenue'] = data['Conversions'] * np.random.uniform(50, 200, n_rows)
    data['ROAS'] = data['Revenue'] / (data['Spend'] + 1)  # Negatively correlated with CPC
    data['CPM'] = (data['Spend'] / data['Impressions'] * 1000)  # Correlated with CPC
    
    # Add some patterns by traffic type
    df = pd.DataFrame(data)
    
    # Paid traffic has higher CPC and lower ROAS
    paid_mask = df['Traffic_Type'] == 'Paid'
    df.loc[paid_mask, 'CPC'] *= 1.5
    df.loc[paid_mask, 'ROAS'] *= 0.7
    
    # Email has higher conversion rate
    email_mask = df['Traffic_Type'] == 'Email'
    df.loc[email_mask, 'Conversions'] *= 1.8
    df.loc[email_mask, 'ROAS'] *= 1.4
    
    # Organic has lower costs
    organic_mask = df['Traffic_Type'] == 'Organic'
    df.loc[organic_mask, 'CPC'] *= 0.6
    df.loc[organic_mask, 'ROAS'] *= 1.3
    
    # ===== CREATE SIMPSON'S PARADOX =====
    # Create "Quality Score" that has opposite correlation in aggregate vs groups
    # Within each device: higher Spend → higher Quality Score (positive correlation)
    # But overall: higher Spend → lower Quality Score (negative correlation)
    # This happens because Desktop has high spend but low quality, Mobile has low spend but high quality
    
    df['Quality_Score'] = 50 + np.random.normal(0, 10, n_rows)
    
    # Desktop: High spend, but positive spend-quality correlation within group
    desktop_mask = df['Device'] == 'Desktop'
    df.loc[desktop_mask, 'Spend'] *= 1.8  # Desktop users spend more
    df.loc[desktop_mask, 'Quality_Score'] = 40 + df.loc[desktop_mask, 'Spend'] * 0.003 + np.random.normal(0, 5, desktop_mask.sum())
    
    # Mobile: Low spend, but positive spend-quality correlation within group  
    mobile_mask = df['Device'] == 'Mobile'
    df.loc[mobile_mask, 'Spend'] *= 0.6  # Mobile users spend less
    df.loc[mobile_mask, 'Quality_Score'] = 60 + df.loc[mobile_mask, 'Spend'] * 0.003 + np.random.normal(0, 5, mobile_mask.sum())
    
    # Tablet: Medium spend, positive correlation
    tablet_mask = df['Device'] == 'Tablet'
    df.loc[tablet_mask, 'Spend'] *= 1.0
    df.loc[tablet_mask, 'Quality_Score'] = 55 + df.loc[tablet_mask, 'Spend'] * 0.003 + np.random.normal(0, 5, tablet_mask.sum())
    
    # This creates Simpson's Paradox:
    # - Within each device type: Spend and Quality_Score are positively correlated
    # - Overall: Spend and Quality_Score appear negatively correlated
    # (because high-spend Desktop has low quality, low-spend Mobile has high quality)
    
    # Add another Simpson's Paradox example with Traffic Type
    # Create "Engagement_Rate" metric
    df['Engagement_Rate'] = 0.0  # Initialize as float
    
    # Within each traffic type: higher CTR → higher Engagement
    # But overall: higher CTR → lower Engagement (because traffic types with high CTR have low base engagement)
    for traffic_type in traffic_types:
        mask = df['Traffic_Type'] == traffic_type
        if traffic_type == 'Paid':
            # Paid has high CTR but low base engagement
            df.loc[mask, 'CTR'] *= 1.3
            base_engagement = 20
        elif traffic_type == 'Email':
            # Email has medium CTR, high base engagement
            df.loc[mask, 'CTR'] *= 1.0
            base_engagement = 60
        elif traffic_type == 'Organic':
            # Organic has low CTR, high base engagement  
            df.loc[mask, 'CTR'] *= 0.8
            base_engagement = 55
        elif traffic_type == 'Social':
            # Social has medium CTR, medium engagement
            df.loc[mask, 'CTR'] *= 1.1
            base_engagement = 40
        else:  # Direct
            df.loc[mask, 'CTR'] *= 0.9
            base_engagement = 50
        
        # Within group: positive correlation between CTR and Engagement
        df.loc[mask, 'Engagement_Rate'] = base_engagement + df.loc[mask, 'CTR'] * 2 + np.random.normal(0, 5, mask.sum())
        # Within group: positive correlation between CTR and Engagement
        df.loc[mask, 'Engagement_Rate'] = base_engagement + df.loc[mask, 'CTR'] * 2 + np.random.normal(0, 5, mask.sum())
    
    # Ensure Quality_Score and Engagement_Rate are within reasonable bounds
    df['Quality_Score'] = df['Quality_Score'].clip(0, 100)
    df['Engagement_Rate'] = df['Engagement_Rate'].clip(0, 100)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    
    # Save to file
    df.to_csv(filename, index=False)
    print(f"✅ Generated sample dataset: {filename}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"\n📊 Dimensions: {', '.join(df.select_dtypes(include=['object']).columns)}")
    print(f"📈 Metrics: {', '.join(df.select_dtypes(include=[np.number]).columns)}")
    print(f"\n🚨 BUILT-IN SIMPSON'S PARADOXES:")
    print(f"   1. Spend vs Quality_Score by Device:")
    print(f"      - Within each device: positive correlation")
    print(f"      - Overall: negative correlation (paradox!)")
    print(f"   2. CTR vs Engagement_Rate by Traffic_Type:")
    print(f"      - Within each traffic type: positive correlation")
    print(f"      - Overall: negative/weak correlation (paradox!)")
    
    return df


if __name__ == "__main__":
    generate_sample_data()
