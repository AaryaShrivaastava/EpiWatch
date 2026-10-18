# Environmental Risk Report
    
## Variables
['temperature_c', 'rainfall_mm', 'humidity_percent', 'air_quality_index', 'water_accumulation_index']

## ML Comparison
region,disease,model,training_period,test_period,MAE,RMSE,R2,sample_count
Bhopal,Dengue,Historical Mean (Baseline),<2024-01-01,>=2024-01-01,4.417957594886728,5.444995440896757,-0.06558172830142928,731
Bhopal,Dengue,RandomForest (Env Features),<2024-01-01,>=2024-01-01,4.1562337737548996,5.243260731491527,0.011914257183424093,731


## Latest Risk Explanation
Region: Bhopal
Disease: Dengue

Environmental Risk Indicator: 36/100

Contributing signals:
- Rainfall is considered in the anomaly score (14d sum: 21.8).
- Humidity is considered in the anomaly score (14d mean: 57.5%).
- Temperature is considered in the anomaly score (14d mean: 18.8°C).

Limitations:
- These associations do not establish causation.
- The underlying dataset may include synthetic observations.
- The score is not a clinically validated risk estimate.
