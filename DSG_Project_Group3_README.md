README FILE GROUP3 DSG  

Enhancer Z-score Prediction using Sequence Embeddings
========================================================

This repository contains a comprehensive Jupyter notebook (`DSG_Project_Group3_CODE.ipynb`) that implements a complete pipeline to predict enhancer Z-scores using DNA sequence data.
The pipeline includes data loading, preprocessing, embedding generation using BioBERT, and model training using a variety of machine learning regressors, followed by visualization and evaluation.

 Project Objective
--------------------
The primary objective of this project is to build a predictive model for enhancer activity, represented by **Z-score**, from raw DNA sequences.
Enhancers are regulatory genomic elements that can upregulate gene expression, and quantifying their strength has important implications in genomics, cancer biology, and gene regulation studies.


 File Descriptions
--------------------
| File                          | Description |
|------------------------------|-------------|
| `DSG_Project_Group3_CODE.ipynb`                  | Jupyter notebook implementing the end-to-end analysis. |
| `Enhancers_zscores_peakscores.csv` | Raw input data: enhancer sequences with associated metadata and Z-scores. |
| `embeddings.csv`             | BioBERT-generated embeddings of each DNA sequence. |
| `model_evaluation_results.csv` | Final performance summary for each model. |

 Dataset Fields
-----------------
The input dataset (`Enhancers_zscores_peakscores.csv`) includes:
- `sequence`: DNA sequence string (e.g., "ACGTGGCT...")
- `Z_score`: Enhancer strength or activity metric (numeric)
- Additional columns: Peak information, coordinates, and experimental scores.

 Cell-by-Cell Breakdown of `DSG_Project_Group3_CODE.ipynb`
----------------------------------------

🔹 **Cell 1: Load Enhancer Dataset**
```python
df2 = pd.read_csv("Enhancers_zscores_peakscores.csv", header=2)
```
- Reads the main enhancer dataset, skipping the first two header rows.
- Prints the shape and column names.
- Verifies correct loading of enhancer sequences and Z-scores.

🔹 **Cell 2: Load Embedding Dataset**
```python
df = pd.read_csv("embeddings.csv", header=2)
```
- Loads previously generated BioBERT embeddings (if available).
- Used in downstream model training without regenerating embeddings.

🔹 **Cell 3: One-Hot Encoding of Sequences**
```python
def one_hot_encode_seq(seq):
    ...
df['onehot_seq'] = df['sequence'].apply(one_hot_encode_seq)
```
- Converts sequences to simple binary vectors (one-hot encoded).
- Provides a traditional ML-compatible representation of sequence data.
- Useful as a comparative baseline to the transformer embeddings.

🔹 **Cell 4: Generate BioBERT Embeddings**
```python
from transformers import AutoTokenizer, AutoModel
...
df['biobart_embedding'] = df['sequence'].progress_apply(get_biobart_embedding)
```
- Loads `biobert-base-cased-v1.1` from Hugging Face.
- Tokenizes each DNA sequence, runs it through BioBERT.
- Extracts hidden states, averages them to get fixed-size embedding.
- Saves embedding-rich dataset as both JSON and CSV formats.

🔹 **Cell 5: LazyPredict Benchmarking**
```python
from lazypredict.Supervised import LazyRegressor
...
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
```
- Uses LazyPredict to train and evaluate many regressors in one go.
- Measures R² and RMSE scores.
- Helps identify the best-performing algorithms without much tuning.

🔹 **Cell 6: Manual Model Training and Evaluation**
```python
models = {
    "ExtraTreesRegressor": ..., "MLPRegressor": ...
}
```
- Trains six carefully selected regressors with proper preprocessing.
- Scales input features where required (MLP, KNN).
- Measures model performance using:
  - R² score (variance explained)
  - RMSE (prediction error)
- Stores results in a summary DataFrame.

🔹 **Cell 7: Save Model Results**
```python
results_df.to_csv("model_evaluation_results.csv")
```
- Saves the evaluation table generated in Cell 6.
- Useful for reporting, reproducibility, and versioning.

🔹 **Cell 8: Actual vs. Predicted Scatter Plots**
```python
plt.scatter(y_test, y_pred)
```
- For each model, plots predicted Z_scores against actual values.
- Evaluates goodness of fit visually.
- Detects underfitting or overfitting trends.

🔹 **Cell 9: Feature Importance (Decision Tree)**
```python
feature_importances = model.feature_importances_
```
- Computes importance of each embedding dimension.
- Ranks top 10 features influencing model predictions.
- Visualizes feature importances using a horizontal bar plot.

🔹 **Cell 10: Dataset Overview**
```python
df.info(), df.describe(), df.columns
```
- Provides data summary including null counts, datatypes, and statistics.
- Checks consistency before ML processing.

🔹 **Cell 11: Histogram of Z-scores**
```python
sns.histplot(df['Z_score'], bins=100, kde=True)
```
- Plots distribution of Z_scores to inspect normality, skewness, and spread.
- Important for selecting appropriate regression strategies.

🔹 **Cell 12: Z-score Boxplot**
```python
sns.boxplot(x=df['Z_score'])
```
- Detects outliers visually using a boxplot.
- Guides data preprocessing decisions (e.g., capping/extending ranges).

 Metrics Used
---------------
| Metric      | Meaning |
|-------------|-------------------------------------------------------------|
| R² Score    | Proportion of variance in Z_score explained by the model.  |
| RMSE        | Standard deviation of prediction errors. Lower is better.  |

 Dependencies
---------------
Install all required packages with:

```
pip install pandas numpy scikit-learn torch transformers lazypredict tqdm seaborn matplotlib
```

>  BioBERT embedding generation is time- and resource-intensive. Use a GPU for best results.

 Usage Instructions
---------------------
1. Download or clone this repository and open `DSG_Project_Group3_CODE.ipynb`.
2. Ensure all required CSV files are present in the correct paths.
3. Install the required Python libraries listed above.
4. Launch Jupyter Notebook or VSCode and run the notebook top to bottom.
5. Review embedding generation, model outputs, visualizations, and saved results.

 Applications & Future Enhancements
-------------------------------------
- Apply to other regulatory DNA regions (e.g., promoters, silencers).
- Use additional biological features like chromatin state or methylation.
- Perform binary classification on enhancer activity (active vs inactive).
- Add model interpretability techniques like SHAP or LIME.

\