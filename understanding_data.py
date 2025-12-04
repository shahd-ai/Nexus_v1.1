import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
from IPython.display import display

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


# =========================================================
# DATASET LIST
# =========================================================
DATASET_CATALOG = {
    "Boston Housing": {"file": "Boston_housing.csv"},
    "Parkinsons": {"file": "parkinsons.csv"},
    "Sonar": {"file": "sonar.csv"},
    "White Wine": {"file": "winequality-white.csv"},
    "Red Wine": {"file": "winequality-red.csv"},
    "Ozone": {"file": "ozone.csv"},
    "Concrete": {"file": "concrete_Data.csv"},
    "Breast Cancer": {"file": "breast_cancer.csv"},
    "Auto MPG": {"file": "auto-mpg.csv"},
    "Bank": {"file": "bank.csv"},
    "Paddy": {"file": "paddydataset.csv"},
}



# =========================================================
# 1️⃣ Detect Target Column
# =========================================================
def detect_target_column(df):
    target_keywords = ["quality", "target", "label", "class", "y", "output", "diagnosis"]

    # Keyword detection
    for col in df.columns:
        if any(key in col.lower() for key in target_keywords):
            return col

    # If last column is categorical
    if df.iloc[:, -1].nunique() < 20:
        return df.columns[-1]

    # If only one category column
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) == 1:
        return cat_cols[0]

    return df.columns[-1]



# =========================================================
# 2️⃣ Target Distribution
# =========================================================
def get_target_distribution(y, task_type):
    if y is None:
        return "Target not found"

    try:
        if task_type == "regression" and pd.api.types.is_numeric_dtype(y):
            return f"Mean: {y.mean():.2f}, Std: {y.std():.2f}"

        counts = y.value_counts(normalize=True)
        return ", ".join([f"{cls}: {pct:.0%}" for cls, pct in counts.items()])

    except Exception as e:
        return f"Error: {str(e)}"



# =========================================================
# 3️⃣ PCA — ONE STANDALONE FUNCTION
# =========================================================
def run_pca(df, target_col=None, dataset_name="dataset"):
    """
    PCA directly from a pandas dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataset already loaded in memory.
    target_col : str
        Target column name.
    dataset_name : str
        Name for saving plots.
    """

    print(f"📌 Running PCA for: {dataset_name}")
    print(f"📊 Shape: {df.shape}")

    # --- Target handling ---
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        print(f"🎯 Target column: {target_col}")
    else:
        y = None
        X = df.copy()
        print("⚠ No target column provided or not found.")

    # --- Select numeric columns ---
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        print("❌ Not enough numeric columns for PCA.")
        return
    
    X_num = X[numeric_cols]

    # --- Standardization ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    # --- PCA ---
    pca = PCA(n_components=min(3, len(numeric_cols)))
    components = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    # --- Save folder ---
    folder = f"plots/{dataset_name}/pca"
    os.makedirs(folder, exist_ok=True)

    # --- Scree plot ---
    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
    plt.title(f"Scree Plot – {dataset_name}")
    plt.xlabel("Component")
    plt.ylabel("Explained variance")
    plt.grid(True)
    plt.savefig(f"{folder}/scree_plot.png")
    plt.show()

    # --- PCA 2D ---
    if components.shape[1] >= 2:
        plt.figure(figsize=(7,6))
        if y is not None:
            plt.scatter(components[:,0], components[:,1], c=y, cmap="viridis", alpha=0.8)
            plt.colorbar(label=target_col)
        else:
            plt.scatter(components[:,0], components[:,1], alpha=0.8)

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA 2D – {dataset_name}")
        plt.grid(True)
        plt.savefig(f"{folder}/pca_2d.png")
        plt.show()

    print(f"📁 PCA plots saved in: {folder}")

    return components, explained_variance


# =========================================================
# 4️⃣ Visualization (Hist, Boxplot, Heatmap)
# =========================================================
def generate_visualizations(name, X, y):
    folder = f"plots/{name.replace(' ', '_')}"
    os.makedirs(folder, exist_ok=True)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    non_constant = [c for c in numeric_cols if X[c].nunique() > 1]

    if not non_constant:
        return

    # Histograms
    for col in non_constant:
        try:
            plt.figure()
            X[col].hist(bins=30)
            plt.title(f"{name} - Histogram {col}")
            plt.savefig(f"{folder}/hist_{col}.png")
            plt.close()
        except:
            pass

    # Boxplots
    for col in non_constant:
        try:
            plt.figure()
            sns.boxplot(x=X[col])
            plt.title(f"{name} - Boxplot {col}")
            plt.savefig(f"{folder}/box_{col}.png")
            plt.close()
        except:
            pass

    # Heatmap
    try:
        corr = X[non_constant].corr()
        if corr.shape[0] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm")
            plt.title(f"{name} - Correlation Heatmap")
            plt.savefig(f"{folder}/heatmap.png")
            plt.close()
    except:
        pass



# =========================================================
# 5️⃣ Dataset Cleaner
# =========================================================
def clean_dataset(df):
    cleaning_report = {}

    # Empty columns
    empty_cols = [col for col in df.columns if df[col].isnull().sum() == df.shape[0]]
    df = df.drop(columns=empty_cols)
    cleaning_report["Empty Columns"] = empty_cols

    # Empty rows
    empty_rows = df[df.isnull().all(axis=1)].index.tolist()
    df = df.drop(index=empty_rows)
    cleaning_report["Empty Rows"] = empty_rows

    # Fill numeric missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    for col in df.select_dtypes(exclude=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df, cleaning_report



# =========================================================
# 6️⃣ Display Plots
# =========================================================
def display_dataset_plots(dataset_name, base_folder="plots"):
    folder_name = dataset_name.replace(" ", "_")
    dataset_path = os.path.join(base_folder, folder_name)
    
    if not os.path.exists(dataset_path):
        print(f"❌ No folder found for: {dataset_name}")
        return
    
    print("="*80)
    print(f" 📊 Plots for: {dataset_name}")
    print("="*80)

    images = [f for f in os.listdir(dataset_path) if f.lower().endswith((".png", ".jpg"))]

    if not images:
        print("⚠ No plots found.")
        return

    for img_name in images:
        img_path = os.path.join(dataset_path, img_name)
        print(f"\n➡ {img_name}")
        display(Image.open(img_path))



# =========================================================
# 7️⃣ MAIN INSPECT FUNCTION
# =========================================================
def inspect_datasets():
    report_data = []

    print("="*80)
    print(" PHASE 2 : DATA UNDERSTANDING (AUTO TARGET + VISUALS + PCA) ")
    print("="*80)

    for name, cfg in DATASET_CATALOG.items():
        try:
            file = cfg["file"]
            if not os.path.exists(file):
                raise FileNotFoundError(file)

            df = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)

            df, cleaning = clean_dataset(df)

            target = detect_target_column(df)
            y = df[target]
            X = df.drop(columns=[target])

            task = "regression" if pd.api.types.is_numeric_dtype(y) else "classification"

            # Info Report
            report_data.append({
                "Dataset": name,
                "Target": target,
                "Task": task,
                "Rows": df.shape[0],
                "Columns": df.shape[1],
                "Missing": df.isnull().sum().sum(),
                "Target Info": get_target_distribution(y, task)
            })

            # VISUALS
            generate_visualizations(name, X, y)

            # PCA (single standalone function)
            run_pca(name, X, y)

            print(f"✔ {name} processed")

        except Exception as e:
            print(f"❌ ERROR in {name}: {e}")

    report = pd.DataFrame(report_data)
    print("\n" + "="*80)
    print(" DATA UNDERSTANDING REPORT ")
    print("="*80)
    print(report.to_string(index=False))

    report.to_csv("understanding.csv", index=False)
    print("\n✔ understanding.csv saved!")
    print("✔ All plots saved in /plots/ folder.")



# =========================================================
# 8️⃣ MAIN
# =========================================================
if __name__ == "__main__":
    inspect_datasets()
