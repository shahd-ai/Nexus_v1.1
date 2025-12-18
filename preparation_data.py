import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def impute_missing_with_rf(df):
    """
    Impute missing values in a DataFrame using Random Forest models.
    Numerical columns → RandomForestRegressor
    Categorical columns → RandomForestClassifier

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with missing values.

    Returns
    -------
    df_filled : pandas.DataFrame
        DataFrame with missing values filled.
    """
    df = df.copy()
    # First pass: fill obvious NaN in numeric columns with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Second pass: fill NaN in categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    


    
    # Identify columns with missing values
    columns_with_na = df.columns[df.isna().sum() > 0]
    
    for col in columns_with_na:
        print(f"Processing column: {col}")
        
        # Split known and missing
        df_known = df[df[col].notna()]
        df_missing = df[df[col].isna()]
        
        if df_missing.empty:
            continue
        
        # Features: all other columns
        X_known = df_known.drop(col, axis=1)
        y_known = df_known[col]
        X_missing = df_missing.drop(col, axis=1)
        
        # Encode categorical variables
        X_full = pd.get_dummies(pd.concat([X_known, X_missing]), drop_first=True)
        X_known_encoded = X_full.iloc[:len(X_known), :]
        X_missing_encoded = X_full.iloc[len(X_known):, :]
        
        # Choose model based on column type
        if df[col].dtype in ['int64', 'float64']:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train and predict
        model.fit(X_known_encoded, y_known)
        predicted_values = model.predict(X_missing_encoded)
        
        # Fill missing values
        df.loc[df[col].isna(), col] = predicted_values
        
        print(f"✅ {col} completed")
    
    print("✅ All columns with NaN have been processed.")
    return df
