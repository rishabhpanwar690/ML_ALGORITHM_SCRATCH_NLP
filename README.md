# ML_ALGORITHM_SCRATCH_NLP
🧠** Custom HR Applicant Tracking System:** 12-Model EnsembleA comprehensive demonstration of 12 custom-built machine learning algorithms, synthesized into a weighted ensemble for predictive applicant screening.

Welcome to the Custom HR Applicant Tracking System (ATS) repository. Rather than relying on standard libraries like scikit-learn for predictive modeling, this project implements 12 distinct machine learning algorithms entirely from scratch using core Python and NumPy.These algorithms are evaluated, weighted, and combined into an ensemble model designed to assess a candidate's suitability based on their interview transcript, resume, and the target job description.

⚙️ System Architecture & WorkflowThe pipeline is designed to handle natural language data and route it through a multi-model decision framework:Data Ingestion: The script reads candidate Transcripts, Resumes, and Job Descriptions from a dataset.Feature Extraction: It concatenates the text and utilizes a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert the textual data into numerical feature arrays.Model Training: 12 distinct algorithms are trained simultaneously on the extracted features.Ensemble Decision: The models vote on the outcome ("Select" or "Reject"), with their votes weighted by their historical accuracy on a hold-out test set.📊 Implemented AlgorithmsThe repository features both regression (interpreting outputs as probabilities) and classification models.

**All underlying mathematics and optimization steps are hard-coded**.
Regression ModelsLinear Regression: Implemented using the Moore-Penrose pseudoinverse.
Ridge Regression: Includes a custom L2 Regularization penalty to prevent overfitting.
KNN Regressor: Distance-based neighbor averaging.
Decision Tree Regressor: Built using variance-reduction splitting criteria.
Random Forest Regressor: An aggregation of bootstrapped decision trees.
Classification ModelsLogistic Regression: Optimized via gradient descent with sigmoid activation.
Gaussian Naive Bayes: Based on probability distributions and prior calculations.
KNN Classifier: Majority voting derived from nearest neighbors.
Perceptron: A foundational neural network methodology.
Linear SVM: Implemented utilizing hinge-loss approximation and gradient descent.
Decision Tree Classifier: Node splitting driven by information gain metrics.
Random Forest Classifier: Ensemble majority voting across randomized trees.

⚖️ **The Weighted Ensemble Mechanism** To maximize predictive accuracy, the system does not rely on a single algorithm. Instead, it utilizes a performance-weighted voting strategy:Evaluate & Rank: Every model is evaluated against a 20% testing split to calculate its baseline accuracy.
Weight Calculation: Each model is assigned a specific weight based on the formula:$w = \max(0.001, \text{accuracy} - 0.5)$Note: Models that perform worse than random chance ($0.5$) are heavily penalized, while highly accurate models are given stronger influence.

The Final Verdict: When evaluating a new applicant, all 12 models generate a prediction. These predictions are multiplied by their respective weights to produce a final Ensemble Score. A score of $\geq 0.5$ results in a "Selected" classification.

🚀 Usage & ImplementationYou can utilize the built-in predict_new_applicant function to evaluate a new candidate profile programmatically.
PYTHON:
from your_script import predict_new_applicant

sample_transcript = "Applicant: I have 5 years of experience in machine learning..."
sample_resume = "Skills: Python, Machine Learning, Data Science, SQL..."
sample_job_description = "Looking for a Senior Data Scientist with 4+ years..."

# Execute the ensemble prediction
decision, score = predict_new_applicant(
    transcript=sample_transcript, 
    resume=sample_resume, 
    job_description=sample_job_description
)

RESULT:========================================
         PREDICTION RESULTS
========================================
Final Decision : Selected
Ensemble Score : 0.6124 (Threshold 0.5)
----------------------------------------
Individual Model Breakdown:
  - Linear Regression        : Select (Raw Score: 0.5821)
  - Gaussian Naive Bayes     : Reject (Raw Score: 0.0000)
  - Random Forest Classifier : Select (Raw Score: 1.0000)
  ... [Displaying all 12 model evaluations]
========================================

**⚠️ Performance Considerations**
Because training 12 complex algorithms from scratch—particularly the Decision Trees and Random Forests—is computationally intensive, the TfidfVectorizer is currently constrained to max_features=100.

This parameter ensures that the script executes within a reasonable timeframe while still demonstrating the fundamental mechanics and viability of the ensemble methodology. If you are running this on a machine with higher computational limits, you may increase this parameter for richer feature extraction.

**🚀 Future Improvements**:
While this repository serves as a robust proof-of-concept for custom-built machine learning algorithms, there are several avenues for future optimization and enhancement to bring the system closer to production-grade standards:

Advanced Feature Engineering & NLP:

Word Embeddings: Transition from a basic TfidfVectorizer to dense vector representations (e.g., Word2Vec, GloVe, or BERT embeddings) to better capture the semantic context of applicant resumes and job descriptions.

Text Preprocessing: Implement robust text normalization pipelines, including lemmatization, stemming, and custom stop-word removal tailored to HR and technical vocabulary.

Algorithmic Optimizations:

Hyperparameter Tuning: Implement a custom Grid Search or Random Search optimization algorithm to dynamically find the best learning rates, tree depths, and penalty terms rather than relying on static defaults.

Advanced Ensembling: Expand the ensemble methodology beyond simple weighted averaging to include Stacking or Boosting techniques (e.g., building a Gradient Boosting mechanism from scratch).

Computational Efficiency:

Parallel Processing: Implement the multiprocessing library to train the 12 algorithms simultaneously across multiple CPU cores, particularly for the Random Forest and Decision Tree models.

Matrix Optimizations: Refactor core mathematical loops using advanced NumPy vectorization or compile heavy mathematical operations using Cython or Numba to decrease training times.

Comprehensive Evaluation Metrics:

Beyond Accuracy: Because HR datasets frequently suffer from class imbalance (many more rejections than selections), rely on more informative metrics. Implement Precision, Recall, F1-Score, and ROC-AUC calculations from scratch to better penalize false positives and false negatives.

Cross-Validation: Build a k-fold cross-validation loop to ensure the models are not overfitting to a specific train_test_split random state.

System Architecture & Deployment:

Model Persistence: Add functionality to save and load the trained weight matrices and tree structures (e.g., via pickle or JSON serialization) so the system does not need to re-train on every execution.

API Integration: Wrap the predict_new_applicant function in a lightweight REST API (using Flask or FastAPI) to allow external systems to query the ATS over HTTP.


<img width="1072" height="621" alt="Screenshot 2026-04-03 215105" src="https://github.com/user-attachments/assets/c281e5f3-007b-4e22-b10c-f6ad18ea3546" />

