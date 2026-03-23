# 🎬 Can We Predict Box Office Success? (IMDb Top 1000)

Everyone assumes that if a movie is highly rated on IMDb, it must have made money. That's not true. *12 Angry Men* has a 9.0 rating and grossed just $4M. This project asks a harder question:

> **Can IMDb metadata alone — ratings, votes, runtime, genre, certificate — predict whether a movie was a Flop, Mid, or Blockbuster?**

No actors. No directors. No marketing spend. Just the structured data available before a movie even releases.

---

## 📦 Dataset

- **Source**: IMDb Top 1000 Movies (Kaggle)
- **Size**: 1000 rows × 16 columns
- **Features used**: `IMDb_Rating`, `Metascore`, `Runtime`, `No_of_Votes`, `Genre`, `Certificate`
- **Dropped**: Title, overview, poster links, actors, directors (to keep features structured and pre-release)

---

## 📊 EDA — What We Explored

### 1. Top 20 by IMDb Rating
The Shawshank Redemption leads at 9.3. Most of the top spots are Drama or Crime films — action rarely dominates here.

### 2. Top 20 by Metascore
Critics and audiences don't always agree. Metascores show meaningful variation from IMDb ratings across the same movies.

### 3. Gross by Certificate (Log Scale)
U and UA-rated films earn dramatically more — wider audiences mean bigger box office. Log scale used because the gap is huge.

### 4. Revenue Over the Decades
Commercial earnings grew steadily until 2020. 2014 and 2018 stand out as particularly strong years. No surprises why 2020 dropped.

### 5. Genre Treemap
Drama dominates the top 1000. Action and Crime follow. If you want critical acclaim, apparently write a Drama.

---

## 🔬 Statistical Tests

Three questions tested properly — not just visually guessed.

### Does rating predict earnings?
```
Pearson r = 0.10 | p-value < 0.05
```
Technically significant, but practically weak. A higher IMDb rating gives you slightly better box office odds — that's it.

### Does genre affect earnings?
```
ANOVA F = 16.69 | p-value ≈ 0.000
```
Yes — strongly. Genre is one of the real drivers of commercial performance.

### Does certificate affect how highly a movie is rated?
```
Chi-Square = 0.00 | p-value = 1.0
```
No relationship at all. A film rated 'A' is just as likely to be loved as one rated 'U'. Quality doesn't care about age restrictions.

---

##  ML — Predicting Box Office Tier

### 🧹 1. Data Cleaning

- `Gross`: Removed `$` and commas → converted to numeric
- `Runtime`: Removed `"min"` → converted to integer
- **16.9% of rows** had missing `Gross` — these were dropped (no target = no use)
- Certificate values standardized: `UA` and `U/A` → unified to single format

---

### 🎯 2. Target Engineering — The Hardest Part

This is where most projects go wrong. A naive approach would split gross into fixed buckets like:
- Under $10M = Flop
- $10M–$100M = Mid
- Over $100M = Blockbuster

**That's wrong.** $10M in 1960 was extraordinary. $10M in 2020 is a flop.

#### The Fix: Year-Relative Quantiles

For each release year, we compute quantile thresholds *within that year's films* — so the definition of "Blockbuster" scales with inflation and industry growth.

| Class | Condition | Count |
|---|---|---|
| **Flop** | Below 50th percentile for that year | 441 |
| **Mid** | Between 50th and 85th percentile for that year | 233 |
| **Blockbuster** | Above 85th percentile for that year | 156 |

This means a film from 1957 grossing $4M could still be classified as a Blockbuster — because relative to its era, it was.

---

### 🔧 3. Feature Engineering

#### Numeric Features
`Released_Year`, `Runtime`, `Metascore`, `IMDb_Rating`
- Missing values → Mean imputation
- Scaled with `StandardScaler`

#### Skewed Feature
`No_of_Votes` (skew = 2.08)
- Missing values → Mean imputation
- Box-Cox power transformation → then `StandardScaler`

#### Categorical Feature
`Certificate`
- Missing values → Most-frequent imputation
- `OneHotEncoder` (handle_unknown='ignore')

#### Multi-label Feature
`Genre` (e.g. `"Action, Crime, Drama"`)
- Custom `GenreBinarizer` transformer built from scratch
- Produces **25+ binary flags** (one per genre)
- Handles multi-genre strings cleanly without leakage

---

### 🔗 4. Pipeline Architecture

Everything is wired into a single sklearn `Pipeline` — no manual transformations, no data leakage between train and test.

```
Pipeline
└── ColumnTransformer
    ├── num    → SimpleImputer → StandardScaler                           [Year, Runtime, Metascore, Rating]
    ├── cat    → SimpleImputer → OneHotEncoder                            [Certificate]
    ├── genre  → GenreBinarizer                                            [Genre]
    └── skew   → SimpleImputer → PowerTransformer(Box-Cox) → StandardScaler  [No_of_Votes]
└── Classifier (swappable)
```

Benefits:
- Clean `fit()` / `predict()` on raw data
- No risk of test data contaminating transformations
- Easily swap classifiers for comparison

---

### 🏆 5. Models & Results

All accuracy ranges reflect natural variation across runs (averaged).

| Metric | Softmax Regression | Decision Tree (Optuna) | XGBoost (Optuna) |
|---|---|---|---|
| Test Accuracy | 70–71% | 66–67% | 65% |
| Macro F1 | 0.65 | 0.58 | 0.58 |
| Flop F1 | 0.79 | 0.76 | 0.78 |
| Mid F1 | 0.60 | 0.53 | 0.51 |
| Blockbuster F1 | 0.61 | 0.53 | 0.46 |

---

### 📌 Why XGBoost Didn't Win

Tree-based models usually dominate tabular data. Not here — and that's actually an insight:

- IMDb features are **linear in nature** (ratings, votes, runtime scale predictably)
- The dataset is small (~830 rows after cleaning) — not enough for trees to find deep patterns
- Class imbalance hits tree models harder than regularized linear models

---

## ⚠️ Important Notes

- **Class imbalance is real and expected** — blockbusters are rare by definition. Macro F1 is used specifically to avoid inflating accuracy on the Flop majority class.
- **Not all top-rated movies are rich** — *12 Angry Men* (9.0 rating, $4M gross) is in this dataset. Critical acclaim ≠ commercial success.
- **Actors and directors were excluded intentionally** — the goal was structured, pre-release features only. Star power via NLP-based reputation scores is a natural next step.

---

## 📉 Limitations

- Budget, marketing spend, screen count, and release timing — the real box office drivers — are not in this dataset
- Only 156 Blockbuster examples limits the model's ability to learn that class well
- Cast and director influence is real but requires NLP to model properly

---

## 🛠️ Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `optuna` · `matplotlib` · `seaborn` · Jupyter · GitHub

---

## 👤 Author

Swayam Prajapati

CSE student exploring data science.  
Feedback, forks, and ⭐ stars are always welcome.
