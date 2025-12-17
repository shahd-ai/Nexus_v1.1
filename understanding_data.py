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


def detect_target_column(df):
    """
    Automatically detect target column using heuristic rules.
    """

    target_keywords = ["quality" , "target", "label", "Class", "y", "output", "diagnosis"]

    for col in df.columns:
        if any(key in col.lower() for key in target_keywords):
            return col

    if df.iloc[:, -1].nunique() < 20:
        return df.columns[-1]

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) == 1:
        return cat_cols[0]

    return df.columns[-1]



def get_target_distribution(y, task_type):
    if y is None:
        return "Target not found"

    try:
        is_numeric = pd.api.types.is_numeric_dtype(y)

        if task_type == "regression" and is_numeric:
            return f"Mean: {y.mean():.2f}, Std: {y.std():.2f}"

        
        counts = y.value_counts(normalize=True)
        return ", ".join([f"{cls}: {pct:.0%}" for cls, pct in counts.items()])

    except Exception as e:
        return f"Error: {str(e)}"



def generate_visualizations(name, X, y):
    folder = f"plots/{name.replace(' ', '_')}"
    os.makedirs(folder, exist_ok=True)

    numeric_cols = X.select_dtypes(include=[np.number]).columns

    
    if len(numeric_cols) == 0:
        print(f"⚠ No numeric columns for {name}, skipping plots.")
        return

    
    non_constant_cols = [col for col in numeric_cols if X[col].nunique() > 1]

    if len(non_constant_cols) == 0:
        print(f"⚠ All numeric columns constant for {name}, skipping plots.")
        return

   
    for col in non_constant_cols:
        try:
            plt.figure()
            X[col].hist(bins=30)
            plt.title(f"{name} - Histogram of {col}")
            plt.savefig(f"{folder}/hist_{col}.png")
            plt.close()
        except:
            pass

    
    for col in non_constant_cols:
        try:
            plt.figure()
            sns.boxplot(x=X[col])
            plt.title(f"{name} - Boxplot of {col}")
            plt.savefig(f"{folder}/box_{col}.png")
            plt.close()
        except:
            pass

   
    try:
        corr = X[non_constant_cols].corr()
        if corr.shape[0] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm")
            plt.title(f"{name} - Correlation Heatmap")
            plt.savefig(f"{folder}/heatmap.png")
            plt.close()
    except:
        print(f"⚠ Heatmap failed for {name}.")


def clean_dataset(df):
    """
    Clean missing values in a dataset:
    - Drop fully empty columns
    - Drop fully empty rows
    - Fill numeric NaN with median
    - Fill categorical NaN with mode
    """

    cleaning_report = {}

    empty_cols = [col for col in df.columns if df[col].isnull().sum() == df.shape[0]]
    df = df.drop(columns=empty_cols)
    cleaning_report["Dropped Empty Columns"] = empty_cols

    empty_rows = df[df.isnull().all(axis=1)].index.tolist()
    df = df.drop(index=empty_rows)
    cleaning_report["Dropped Empty Rows"] = len(empty_rows)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    cleaning_report["Numeric Filled"] = {
        col: int(df[col].isnull().sum()) for col in num_cols
    }

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    cleaning_report["Categorical Filled"] = {
        col: int(df[col].isnull().sum()) for col in cat_cols
    }

    return df, cleaning_report


def display_dataset_plots(dataset_name, base_folder="plots"):
    folder_name = dataset_name.replace(" ", "_")
    dataset_path = os.path.join(base_folder, folder_name)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Aucun dossier trouvé pour : {dataset_name}")
        print(f"Chemin recherché : {dataset_path}")
        return
    
    print("="*80)
    print(f" 📊 Visualisations pour le dataset : {dataset_name}")
    print("="*80)

    images = [f for f in os.listdir(dataset_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    if not images:
        print("⚠ Aucun plot trouvé dans ce dossier.")
        return

    for img_name in images:
        img_path = os.path.join(dataset_path, img_name)
        print(f"\n➡ Plot : {img_name}")
        display(Image.open(img_path))

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

    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        print(f"🎯 Target column: {target_col}")
    else:
        y = None
        X = df.copy()
        print("⚠ No target column provided or not found.")

    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        print("❌ Not enough numeric columns for PCA.")
        return
    
    X_num = X[numeric_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    pca = PCA(n_components=min(3, len(numeric_cols)))
    components = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    folder = f"plots/{dataset_name}/pca"
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
    plt.title(f"Scree Plot – {dataset_name}")
    plt.xlabel("Component")
    plt.ylabel("Explained variance")
    plt.grid(True)
    plt.savefig(f"{folder}/scree_plot.png")
    plt.show()

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

    print(f" PCA plots saved in: {folder}")

    return components, explained_variance

def inspect_datasets():
    report_data = []

    print("="*80)
    print(" PHASE 2 : DATA UNDERSTANDING (AUTO TARGET + VISUALS) ")
    print("="*80)

    for name, cfg in DATASET_CATALOG.items():
        try:
            file = cfg["file"]

            if not os.path.exists(file):
                raise FileNotFoundError(file)

            
            if file.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            target_col = detect_target_column(df)
            y = df[target_col]
            X = df.drop(columns=[target_col])

            task_type = "regression" if pd.api.types.is_numeric_dtype(y) else "classification"

            n_samples, n_features = X.shape
            missing = df.isnull().sum().sum()
            missing_pct = (missing / df.size) * 100

            num_cols = X.select_dtypes(include=[np.number]).shape[1]
            cat_cols = X.select_dtypes(exclude=[np.number]).shape[1]

            target_info = get_target_distribution(y, task_type)

            report_data.append({
                "Dataset": name,
                "Target Column": target_col,
                "Task": task_type,
                "Rows": n_samples,
                "Columns": n_features,
                "Numeric Cols": num_cols,
                "Categorical Cols": cat_cols,
                "Missing": f"{missing} ({missing_pct:.1f}%)",
                "Target Info": target_info
            })

            generate_visualizations(name, X, y)

            print(f"✔ Processed: {name}")

        except Exception as e:
            print(f"❌ ERROR - {name}: {e}")

    report = pd.DataFrame(report_data)
    print("\n" + "="*80)
    print(" DATA UNDERSTANDING REPORT ")
    print("="*80)
    print(report.to_string(index=False))

    report.to_csv("understanding.csv", index=False)
    print("\n✔ Report saved as data_understanding.csv")
    print("✔ All visuals saved under /plots/")



if __name__ == "__main__":
    inspect_datasets()
