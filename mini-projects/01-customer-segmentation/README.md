# Project 1: Customer Segmentation Analysis

##  Objective
Segment customers based on purchasing behavior using RFM analysis and K-Means clustering.

##  Dataset
- **Source**: UCI Online Retail Dataset
- **Size**: 541,909 transactions
- **Period**: Dec 2010 - Dec 2011
- **Customers**: 4,338 unique customers

##  Technologies Used
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- Algorithms: K-Means Clustering, PCA
- Platform: Google Colab

##  Methodology
1. **Data Cleaning**: Removed nulls, cancellations, negative values
2. **Feature Engineering**: Created RFM metrics
3. **Clustering**: Applied K-Means (k=4)
4. **Dimensionality Reduction**: PCA for visualization
5. **Analysis**: Generated business insights

##  Key Results
- **Silhouette Score**: 0.375
- **4 Customer Segments Identified**:
  - Champions: 2.8% of customers
  - At Risk: 56.5% of customers
  - New Customers: 17.2% of customers
  - Potential Loyalists: 23.6% of customers

##  Files
- `01_Customer_Segmentation.ipynb` - Main analysis notebook
- `customer_segments.csv` - Segmented customer data
- `01_Customer_Segmentation.html` - HTML version for viewing


##  Business Recommendations
- **Champions**: VIP programs, early access to products
- **At Risk**: Win-back campaigns, special offers
- **New Customers**: Welcome series, onboarding
- **Potential Loyalists**: Loyalty program, personalized recommendations

##  Visualizations
- RFM distribution plots
- Cluster profile comparison
- 2D PCA scatter plot
- Interactive 3D cluster visualization


