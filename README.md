# Emotion-Aware Toxicity Detection

### Deep Learning Project — Cornell Tech  
**Contributors:** Scott Pitcher (sp2668), Kendall Miller (kkm88), Maria Miskaryan (mm3555)  
**Course:** Deep Learning (CS5785)  

---

## 📖 Overview
This project explores whether **emotion recognition can enhance toxicity detection** on social media platforms.  
While most toxicity classifiers rely solely on syntactic or lexical cues, emotionally charged but non-toxic content often causes false positives.  
We introduce an **emotion-aware multi-task learning framework** that jointly trains on toxicity and emotion detection to improve nuance, interpretability, and robustness in toxicity classification.

---

## 🎯 Research Questions
1. Can shared emotional representations in a multi-task framework improve nuanced toxicity detection?  
2. Which emotions (e.g. anger, fear, sadness) correlate most strongly with toxicity?  
3. Are there confounders or shared token groups across the emotion and toxicity tasks?  
4. How does interpretability differ when training tasks separately versus jointly?  
5. Does grouping semantically similar terms improve interpretability?  
6. Are there any learned “shortcuts” the model uses when making predictions?

---

## 🧠 Model Architecture
We build upon **BERT** and its variants (e.g., DistilBERT) using **Hugging Face Transformers** (PyTorch).  

- **Shared Encoder:** Pre-trained BERT backbone.  
- **Dual Heads:**  
  - **Toxicity classification head** — trained on the Jigsaw Toxic Comment dataset.  
  - **Emotion classification head** — trained on GoEmotions (or SemEval).  
- **Training Variants:**
  - **Baseline:** Single-task fine-tuning on toxicity.  
  - **Multi-task:** Joint training on emotion and toxicity.  
  - **Sequential:** Pretraining on emotion, then fine-tuning for toxicity.

Losses are combined via weighted objectives to explore the impact of shared representation learning.

---

## 📊 Datasets
| Dataset | Task | Description | Source |
|----------|------|--------------|---------|
| **Jigsaw Toxic Comment Classification** | Toxicity detection | Multi-label dataset for classifying online toxic comments | [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) |
| **GoEmotions** | Emotion classification | 27-class dataset for fine-grained emotion recognition | [Kaggle](https://www.kaggle.com/datasets/debarshichanda/goemotions) |
| *(Alternative)* SemEval | Emotion classification | Emotion detection dataset from Twitter | [Kaggle](https://www.kaggle.com/datasets/azzouza2018/semevaldatadets) |

---

## ⚙️ Methods & Experiments
- Fine-tune BERT with custom dual-task heads.
- Compare multi-task vs single-task and sequential training strategies.
- Evaluate using **F1-score**, **precision**, **recall**, and **ROC-AUC**.
- Perform **ablation studies** by adjusting loss weightings between tasks.

---

## 🔍 Interpretability Analysis
To explore how emotions influence toxicity understanding:
- Use **attention visualizations**, **LIME**, and **SHAP** to identify key tokens.
- Compare overlap and divergence between emotion and toxicity attention patterns.
- Group semantically similar tokens via **embedding clustering** to improve interpretability.
- Identify model “shortcuts” and token biases.

---

## 💡 Innovation
This project introduces an **emotion-aware, multi-task BERT framework** for toxicity detection —  
allowing the model to differentiate between:
- Emotionally intense but non-toxic content, and  
- Genuinely harmful language.

Such improvements can enhance content moderation systems used by platforms like YouTube or Instagram.

---

## 📈 Evaluation Metrics
- **F1-score**
- **Precision**
- **Recall**
- **ROC-AUC**
- **Token overlap and interpretability scores** across tasks

---

## 📚 References
1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).  
   *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*  
   [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

2. Mnassri, B., Rajapaksha, S., Farahbakhsh, R., & Crespi, N. (2023).  
   *Hate Speech and Offensive Language Detection Using an Emotion-Aware Shared Encoder.*  
   [arXiv:2302.08777](https://arxiv.org/abs/2302.08777)

---

## 🧑‍💻 Authors
- **Scott Pitcher** 
- **Kendall Miller**
- **Maria Miskaryan**

