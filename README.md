# ChatGPT Prompt Analysis and Grade Predictor

This project utilizes advanced machine learning techniques to predict student homework scores based on their interactions with ChatGPT. By examining the text prompts submitted to ChatGPT, this study seeks to uncover the predictive power of student responses, code-writing skills, and prompt characteristics.

---

## Project Overview

The project develops a comprehensive methodology to analyze the nature and content of prompts students send to ChatGPT, correlating these interactions with their academic performance. This involves creating a complex feature set that includes text similarity measures, code response analysis, and traditional text analysis metrics.

---

## Methodology

### Feature Engineering
- **Weighted Targets**: Identified similarities between user prompts and known homework questions, assigning weights (`weighted_target`) based on their importance to the overall score.
- **Code Responses**: Tracked the number of code blocks ChatGPT returned (`code_responses`).
- **Bag of Words**: Implemented a bag-of-words model using the 500 most frequent words from high-scoring students' responses.

### Predictive Modeling
- **Initial Models**: Tested neural networks, random forests, and decision trees without satisfactory results.
- **Feature Enhancement**: Introduced a novel TF-IDF based similarity metric comparing code responses to a standard answer key, significantly improving model performance.
- **Model Selection**: After integrating the new features, the random forest model was selected for its low mean squared error and robust handling of diverse data.

---

## Results

The integration of advanced text analysis and machine learning methods yielded significant insights into the predictive factors of academic success:

- **Random Forest Performance**:
  - **RMSE (Root Mean Square Error)**: 8.55
  - **R² Score**: 0.46

These metrics indicate that our model can reliably predict student grades based on their interactions with ChatGPT, particularly when enhanced by the novel similarity features.

### Key Findings
- The addition of the similarity metric based on the standard answer key was crucial in reducing the RMSE and improving the R² score.
- The predictive model's accuracy highlights the effectiveness of combining traditional and novel data analysis techniques in educational settings.

---

## Team Contributions

- **Ali Fehmi Yıldız**: Focused on adding new features and experimenting with various predictive models.
- **Berkay Yenilmez**: Spearheaded feature enhancement and model testing.
- **Irfaan Bin Ahmad**: Conducted clustering analysis to categorize student interactions.
- **Muhammed Emir İnce**: Key role in feature development and model refinement.

---

## Technologies Used

- **Languages**: Python
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `scikit-learn`
  - `tqdm`, `beautifulsoup4`, `graphviz`

---
