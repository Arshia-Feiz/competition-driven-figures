# Competition-Driven Figures of Merit in Technology Roadmapping

## 📊 Project Overview

This project recreates performance-based competition figures for BMW and Mercedes-Benz based on academic research into competitive dynamics in the automotive industry. The analysis focuses on understanding how competitive pressures drive technology optimization and strategic positioning between luxury car manufacturers.

## 🎯 Key Objectives

- **Competitive Analysis**: Model strategic interactions between BMW and Mercedes-Benz
- **Performance Metrics**: Analyze evolution of key performance indicators
- **Strategic Positioning**: Understand how manufacturers optimize specific metrics
- **Game Theory Application**: Model competitive responses and best strategies

## 🛠️ Technical Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Statistical analysis and outlier detection
- **Jupyter Notebooks**: Interactive analysis

## 📈 Methodology

### 1. Data Preprocessing
- **Dataset**: Large Kaggle automotive dataset
- **Manufacturers**: BMW and Mercedes-Benz focus
- **Time Period**: 1975-2015 analysis window
- **Metrics**: Horsepower, fuel efficiency, acceleration, weight
- **Quality Control**: Percentile-based outlier removal

### 2. Competitive Analysis
- **Best Response Curves**: Model how each manufacturer optimizes metrics
- **Temporal Evolution**: Track performance changes over time
- **Strategic Positioning**: Analyze competitive responses
- **Performance Trends**: Identify optimization patterns

### 3. Game Theory Modeling
- **Payoff Tables**: Multi-dimensional performance comparisons
- **Figures of Merit**: Integrated metrics for strategic evaluation
- **Competitive Dynamics**: Model strategic interactions
- **Technology Evolution**: Understand competitive pressures

## 📊 Results

### Key Visualizations
- **Performance Evolution**: Color-coded scatter plots tracking changes over time
- **Best Response Analysis**: How manufacturers optimize specific metrics
- **Competitive Landscape**: Strategic positioning relative to competitors
- **Technology Trends**: Long-term performance improvement patterns

### Strategic Insights
- **Optimization Patterns**: How manufacturers maximize/minimize specific metrics
- **Competitive Responses**: Strategic reactions to competitor actions
- **Technology Progression**: Understanding competitive pressures on innovation
- **Market Dynamics**: Strategic positioning in luxury automotive segment

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Data Requirements
- Kaggle automotive dataset (1975-2015)
- BMW and Mercedes-Benz vehicle data
- Performance metrics: horsepower, fuel efficiency, acceleration, weight
- Temporal data for trend analysis

### Competitive Analysis Workflow
```python
# Example competitive analysis setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Load and filter data
df = pd.read_csv('automotive_data.csv')
luxury_brands = ['BMW', 'Mercedes-Benz']
filtered_df = df[df['Manufacturer'].isin(luxury_brands)]

# Outlier removal using percentile method
def remove_outliers(df, column, lower_percentile=5, upper_percentile=95):
    lower_bound = df[column].quantile(lower_percentile / 100)
    upper_bound = df[column].quantile(upper_percentile / 100)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Analyze competitive positioning
def analyze_competitive_positioning(df, metric):
    yearly_stats = df.groupby(['Year', 'Manufacturer'])[metric].agg(['mean', 'max', 'min'])
    return yearly_stats
```

## 📁 Project Structure

```
competition-driven-figures/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── automotive_dataset.csv
│   └── processed/
│       ├── luxury_brands_filtered.csv
│       └── competitive_analysis.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_competitive_analysis.ipynb
│   ├── 03_best_response_analysis.ipynb
│   └── 04_game_theory_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── competitive_analysis.py
│   ├── best_response_analysis.py
│   └── game_theory_models.py
├── results/
│   ├── visualizations/
│   │   ├── performance_evolution.png
│   │   ├── best_response_curves.png
│   │   └── competitive_landscape.png
│   └── analysis/
│       ├── payoff_tables.csv
│       └── figures_of_merit.csv
├── documentation/
│   └── (PDF available via Google Drive link)
└── documentation/
    └── methodology_notes.md
```



## 🔬 Research Applications

This project demonstrates:
- **Competitive Intelligence**: Understanding industry dynamics
- **Strategic Analysis**: Supporting technology strategy decisions
- **Game Theory Application**: Modeling competitive interactions
- **Technology Roadmapping**: Informing strategic planning

## 📚 References

- **Original Research**: Academic studies on competitive dynamics in automotive industry
- **Dataset**: [Kaggle Automotive Dataset](https://www.kaggle.com/datasets/CooperUnion/car-dataset)
- **Game Theory**: Strategic interaction modeling in technology markets

## 🎓 Academic Context

This work was conducted at ISAE-SUPAERO as part of research into:
- Competitive dynamics in technology markets
- Strategic technology positioning
- Game theory applications in industry analysis
- Technology roadmapping methodologies

## 👨‍💻 Author

**Arshia Feizmohammady**
- Industrial Engineering Student, University of Toronto
- Research focus: Competitive analysis and strategic technology planning
- [LinkedIn](https://linkedin.com/in/arshiafeiz)
- [Personal Website](https://arshiafeizmohammady.com)

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic or commercial applications.

---

*This project demonstrates the application of competitive analysis and game theory in understanding technology evolution and strategic positioning in the automotive industry.*
