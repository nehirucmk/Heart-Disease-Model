from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def prepare_data(file_path):
    
    df = pd.read_csv(file_path)
    
    # One-Hot Encoding for categorical variables
    # this prevents the model from thinking 3 is 'greater' than 1 in categories
    categorical_cols = ['cp', 'slope', 'restecg', 'thal', 'ca']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # splitting Features (X) and Target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # splitting into train and test sets
    # 'stratify=y' to ensure both sets have the same ratio of heart disease
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# testing the function
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = prepare_data('heart.csv')
    print(f"preprocessed data shape: {X_train.shape}")
    print("features after encoding:", list(features))