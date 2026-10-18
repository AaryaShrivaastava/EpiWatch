# Limitations of Environmental Risk Modeling

- **Correlation does not establish causation.** The identified associations between environmental variables (e.g., rainfall, temperature) and epidemiological observations are strictly observational.
- **Delayed and Nonlinear Relationships.** Environmental effects often involve complex, non-linear biological pathways (e.g., mosquito breeding cycles for Dengue) that may not be perfectly captured by simple linear or negative-binomial lags.
- **Seasonal Confounding.** High temperatures or rainfall often correlate with specific times of the year when disease transmission is inherently higher due to other sociodemographic or ecological factors. Disentangling true environmental risk from calendar seasonality remains challenging.
- **Reporting Delays.** Epidemiological observations rely on surveillance systems. Delays or variations in testing and reporting can artificially shift the timing of observed peaks, skewing lag analysis.
- **Synthetic Data Limitations.** Models trained or calibrated on synthetic or source-calibrated proxy data cannot definitively establish real-world epidemiological parameters.
- **Small Datasets.** Statistical parameter estimation (e.g., Negative Binomial) may fail or yield overly wide confidence intervals if sufficient historical outbreak data is unavailable.
- **Risk Score Validation.** The thresholds (Moderate, High, Very High) and component weights used in the environmental risk score are provisional engineering thresholds. They require prospective real-world validation.
- **Not a Medical Diagnosis.** The risk score represents an environmental risk indicator, NOT a medically validated probability that a specific individual will contract the disease or that an outbreak will definitively occur.
- **Future Data Dependency.** Predictive models must only rely on currently observable data. Attempting to forecast future cases using actual future environmental observations (unless provided by a validated meteorological forecast) constitutes temporal leakage.
