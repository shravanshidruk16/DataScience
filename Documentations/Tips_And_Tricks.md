# 🧠 Data Science & Machine Learning Tips Vault  
### By Shravan Shidruk  

> A growing collection of practical tips, shortcuts, debugging tricks, optimization techniques, and workflow improvements discovered through real project experience, failures, and experimentation.

---

## 📌 Why This Repository Exists

During my Data Science and ML journey, I realized:

- Small tricks save **hours of debugging**
- Clean workflow > complex algorithms
- Optimization matters as much as modeling
- Real growth happens through trial and error

This README acts as my **personal engineering cheat-sheet** that others can also use.

---

# 📒 Jupyter Notebook Productivity

### 🔹 Export Notebook as Python Script Without Metadata / Cell Numbers

```python
!jupyter nbconvert --to script your_notebook_name.ipynb
```

**Benefits**
- Removes `In[]` / `Out[]`
- Clean GitHub view
- Production-ready code format
- Easier to review

---

### 🔹 Clear All Outputs Before Sharing Notebook

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb
```

---

### 🔹 Restart Notebook Kernel (Fix Memory Freeze)

```python
import os
os._exit(0)
```

---

### 🔹 Show All Outputs (Avoid Truncated Tables)

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

---

# 📊 Data Visualization Tricks

### 🔹 Save High Quality Plots

```python
plt.savefig("plot.png", dpi=300, bbox_inches="tight")
```

For research papers:

```python
plt.savefig("plot.png", dpi=600)
```

---

### 🔹 Remove Extra White Space

```python
plt.tight_layout()
```

---

### 🔹 Set Custom Figure Size

```python
plt.figure(figsize=(10,6))
```

---

### 🔹 Add Grid to Improve Readability

```python
plt.grid(True)
```

---

# ⚡ Performance Boosting Tricks

### 🔹 Check Memory Usage of DataFrame

```python
df.memory_usage(deep=True)
```

---

### 🔹 Reduce DataFrame Memory Size

```python
for col in df.select_dtypes(include='float'):
    df[col] = df[col].astype('float32')
```

---

### 🔹 Avoid Slow `apply()` — Use Vectorization

```python
df["new_col"] = df["a"] + df["b"]
```

---

### 🔹 Faster CSV Loading

```python
df = pd.read_csv("file.csv", low_memory=False)
```

---

# 🧹 Data Cleaning Hacks

### 🔹 Detect Missing Values

```python
df.isna().sum()
```

---

### 🔹 Drop Duplicate Rows

```python
df.drop_duplicates(inplace=True)
```

---

### 🔹 Handle Outliers Using IQR

```python
Q1 = df["col"].quantile(0.25)
Q3 = df["col"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["col"] >= Q1 - 1.5*IQR) & (df["col"] <= Q3 + 1.5*IQR)]
```

---

### 🔹 Rename Columns Properly

```python
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
```

---

# 🤖 Machine Learning Shortcuts

### 🔹 Proper Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 🔹 Save ML Model

```python
import joblib
joblib.dump(model, "model.pkl")
```

---

### 🔹 Load Saved Model

```python
model = joblib.load("model.pkl")
```

---

### 🔹 Cross Validation

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

---

# 🧪 Debugging Tricks

### 🔹 Show All Columns

```python
pd.set_option('display.max_columns', None)
```

---

### 🔹 Check Data Types

```python
df.dtypes
```

---

### 🔹 Dataset Summary

```python
df.info()
df.describe()
```

---

### 🔹 Find Unique Values

```python
df["col"].unique()
```

---

# 💡 Workflow Tips

- Always keep **raw data untouched**
- Use separate notebooks for:
  - EDA
  - Modeling
  - Final pipeline
- Use virtual environments
- Maintain project structure
- Comment your assumptions
- Track experiments

---

# 🚀 GitHub Friendly Practices

- Convert notebooks to `.py` before pushing
- Add `.gitignore`
- Use `requirements.txt`
- Keep folders organized
- Write README for every project

---

# 🏆 Final Advice

> "Most ML problems are Data Cleaning problems in disguise." <br>
> **_This Tips and Tricks is a living document for my Debugging and Problem Solving Journey of ML.<br>
As I grow, learn, and fail forward — this space evolves with me._**

Keep experimenting. Break things. Fix things. That’s how real engineers grow.<br>
**_DevDiscipline_**
