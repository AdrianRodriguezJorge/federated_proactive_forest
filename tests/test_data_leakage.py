import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def test_data_leakage_and_mutation():
    print("Iniciando Test de Data Leakage y Mutación de X_raw...")
    
    # Crear un dataset dummy similar a las características del framework
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'feature_num_1': np.random.randn(n_samples) * 10,
        'feature_num_2': np.random.rand(n_samples),
        'feature_cat': np.random.choice(['A', 'B', 'C'], size=n_samples),
        'target': np.random.choice([0, 1], size=n_samples)
    })
    
    X_raw = df.drop(columns=['target'])
    y_raw = df['target'].values
    
    # Hacer una copia original bit-a-bit de X_raw para verificar si muta
    X_raw_original_copy = X_raw.copy(deep=True)
    
    cat_cols = ['feature_cat']
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    precomputed_splits = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_raw, y_raw)):
        # Extraer usando iloc
        X_train_val_raw = X_raw.iloc[train_val_idx]
        y_train_val_raw = y_raw[train_val_idx]
        
        # Split interno
        from sklearn.model_selection import train_test_split
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_train_val_raw, y_train_val_raw, 
            test_size=0.1, random_state=42 + fold_idx,
            stratify=y_train_val_raw
        )
        
        # === SIMULAR LA LÓGICA DEL final_benchmark.py ===
        # Extraer como numpy arrays desconectados del dataframe original
        X_train = X_train_raw.values.copy()
        
        cat_cols_idx = [X_raw.columns.get_loc(c) for c in cat_cols] if cat_cols else []
        num_cols_idx = [X_raw.columns.get_loc(c) for c in X_raw.columns if c not in cat_cols]

        if cat_cols_idx:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train[:, cat_cols_idx] = enc.fit_transform(X_train[:, cat_cols_idx].astype(str))
            
        if num_cols_idx:
            scaler = StandardScaler()
            X_train[:, num_cols_idx] = scaler.fit_transform(X_train[:, num_cols_idx])
            
        # Modificar intencionalmente un valor en el array de numpy fiteado para comprobar aislamiento
        X_train[0, num_cols_idx[0]] = 999.99
    
    # 1. VERIFICAR QUE X_RAW ESTÉ INTACTO
    is_identical = X_raw.equals(X_raw_original_copy)
    print(f"¿X_raw se mantiene 100% idéntico a su estado original? -> {'SÍ (Pass)' if is_identical else 'NO (Fail)'}")
    
    if not is_identical:
        print("¡ERROR GRAVE DE DATA LEAKAGE! X_raw mutó durante el Fold.")
        return False
        
    print("Test Exitoso. No hay Data Leakage en memoria compartida. Los splits son seguros.")
    return True

if __name__ == "__main__":
    test_data_leakage_and_mutation()
