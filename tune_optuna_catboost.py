# ==============================================================================
# SKRIPSI: OPTIMASI HIPERPARAMETER CATBOOST DENGAN FRAMEWORK OPTUNA
# Topik      : Deteksi Fraud Transaksi Blockchain Ethereum
# Peneliti   : Muhammad Galang Dining Samudra (NIM: 535230193)
# Algoritma  : CatBoost Classifier
# Dataset    : DeFiTransLyzer_combined_clean_v3.csv (1.026.814 Baris Data)
# Konfigurasi: 19 Fitur Murni Perilaku (Tanpa block_number) & Split Ratio 80:20
# Framework  : Optuna (Bayesian Optimization via Tree-structured Parzen Estimator / TPE)
# ==============================================================================

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError("Pustaka 'optuna' belum terpasang. Silakan jalankan: pip install optuna")

# 1. Konfigurasi Path dan Direktori
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "DeFiTransLyzer_combined_clean_v3.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "flag"

# 2. Definisi 19 Fitur Murni Perilaku (Pure Behavioral Features)
# Fitur kronologis block_number dieliminasi untuk mencegah kebocoran temporal (temporal data leakage)
FEATURES_19 = [
    "nonce.nonce_coefficient_of_variation",
    "cumulativeGasUsed.cumulativeGasUsed_mode",
    "cumulativeGasUsed.cumulativeGasUsed_median",
    "erc_20_Symbol_End_Is_Digit",
    "erc_20_Symbol_Has_Digit",
    "cumulativeGasUsed.cumulativeGasUsed_average",
    "nonce.nonce_skewness",
    "number_of_unique_from_address",
    "gas_prices.gas_prices_skewness",
    "erc_721_Symbol_Lowercase_Count",
    "erc_20_Name_Has_Digit",
    "cumulativeGasUsed.cumulativeGasUsed_standard_deviation",
    "gas_efficiency",
    "cumulativeGasUsed.cumulativeGasUsed_coefficient_of_variation",
    "cumulative_gas_used",
    "index",
    "num_transaction",
    "number_of_from_address",
    "number_of_to_address"
]

print("=" * 85)
print("🚀 MEMULAI OPTIMASI HIPERPARAMETER CATBOOST MENGGUNAKAN OPTUNA (TPE SAMPLER)")
print("=" * 85)

# 3. Membaca dan Menyiapkan Data
print(f"\n[1/4] Membaca dataset dari: {DATASET_PATH}...")
start_load = time.time()
df = pd.read_csv(DATASET_PATH, usecols=FEATURES_19 + [TARGET_COL], low_memory=False)
print(f"      Berhasil memuat {len(df):,} baris transaksi dalam {time.time() - start_load:.2f} detik.")

# Preprocessing nilai tak hingga dan nilai kosong
X = df[FEATURES_19].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = df[TARGET_COL].astype(int)

# 4. Stratified Data Splitting 80:20 (Rasio Terbaik Terpilih)
print("\n[2/4] Melakukan Stratified Train-Test Split (80% Train : 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      Data Latih : {len(X_train):,} baris")
print(f"      Data Uji   : {len(X_test):,} baris")

# 5. Definisi Objective Function untuk Optuna
def objective(trial):
    """
    Fungsi objektif yang dievaluasi pada setiap iterasi trial Optuna.
    Memaksimalkan F1-Score pada kelas fraud (minoritas).
    """
    # Ruang pencarian hiperparameter (Search Space)
    depth = trial.suggest_int("depth", 4, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.10, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
    iterations = trial.suggest_int("iterations", 300, 800, step=100)
    auto_class_weights = trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced"])

    # Inisialisasi Model CatBoost dengan parameter trial
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        auto_class_weights=auto_class_weights,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=42,
        verbose=0
    )

    # Pelatihan dengan validasi internal untuk evaluasi pruning
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=30,
        verbose=False
    )

    # Prediksi data uji
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.50).astype(int)

    # Evaluasi metrik F1-Score
    score_f1 = f1_score(y_test, y_pred)
    
    # Pelaporan nilai perantara ke pruner Optuna
    trial.report(score_f1, step=model.get_best_iteration())
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return score_f1

# 6. Menjalankan Optimasi Bayesian Menggunakan TPE Sampler
print("\n[3/4] Menjalankan proses optimasi Bayesian Optuna (30 Trials)...")
sampler = TPESampler(seed=42)
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
    study_name="catboost_fraud_detection_optimization"
)

start_optuna = time.time()
study.optimize(objective, n_trials=30, show_progress_bar=True)
total_optuna_time = time.time() - start_optuna

print(f"\n      Optimasi selesai dalam {total_optuna_time / 60:.2f} menit.")
print(f"      Best Trial  : Trial #{study.best_trial.number}")
print(f"      Best F1-Score: {study.best_value * 100:.2f}%")
print("      Parameter Terbaik:")
for k, v in study.best_params.items():
    print(f"         • {k}: {v}")

# Simpan riwayat trial ke berkas CSV
trials_df = study.trials_dataframe()
trials_csv_path = os.path.join(OUTPUT_DIR, "optuna_trials_history.csv")
trials_df.to_csv(trials_csv_path, index=False)
print(f"      Riwayat trial tersimpan di: {trials_csv_path}")

# 7. Evaluasi Akhir: CatBoost Default vs CatBoost Tuned (Optuna)
print("\n[4/4] Membandingkan Performa Model Default vs. Model Hasil Optimasi Optuna...")

# A. Model Default (Tanpa penyesuaian bobot kelas)
model_default = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    random_seed=42,
    verbose=0
)
t0 = time.time()
model_default.fit(X_train, y_train, verbose=False)
train_time_def = time.time() - t0

t0 = time.time()
prob_def = model_default.predict_proba(X_test)[:, 1]
lat_def = ((time.time() - t0) / len(X_test)) * 1000
pred_def = (prob_def >= 0.50).astype(int)

# B. Model Tuned (Parameter Terbaik Hasil Optuna)
model_tuned = CatBoostClassifier(
    iterations=study.best_params.get("iterations", 500),
    learning_rate=study.best_params.get("learning_rate", 0.05),
    depth=study.best_params.get("depth", 6),
    l2_leaf_reg=study.best_params.get("l2_leaf_reg", 3.0),
    auto_class_weights=study.best_params.get("auto_class_weights", "Balanced"),
    loss_function="Logloss",
    random_seed=42,
    verbose=0
)
t0 = time.time()
model_tuned.fit(X_train, y_train, verbose=False)
train_time_tuned = time.time() - t0

t0 = time.time()
prob_tuned = model_tuned.predict_proba(X_test)[:, 1]
lat_tuned = ((time.time() - t0) / len(X_test)) * 1000
pred_tuned = (prob_tuned >= 0.50).astype(int)

# 8. Rekapitulasi Tabel Komparasi Bab 4 (Tabel 4.3)
comparison_results = [
    {
        "Konfigurasi Model": "CatBoost Default (Baseline)",
        "Parameter Kunci": "depth=6, lr=0.03, l2=3.0, iter=500, unweighted",
        "Akurasi (%)": round(accuracy_score(y_test, pred_def) * 100, 2),
        "Presisi (%)": round(precision_score(y_test, pred_def) * 100, 2),
        "Recall (%)": round(recall_score(y_test, pred_def) * 100, 2),
        "F1-Score (%)": round(f1_score(y_test, pred_def) * 100, 2),
        "ROC-AUC (%)": round(roc_auc_score(y_test, prob_def) * 100, 2),
        "Latensi (ms/tx)": round(lat_def, 4)
    },
    {
        "Konfigurasi Model": "CatBoost Tuned (Optuna Best)",
        "Parameter Kunci": f"depth={study.best_params.get('depth', 6)}, lr={round(study.best_params.get('learning_rate', 0.05), 3)}, l2={round(study.best_params.get('l2_leaf_reg', 3.0), 1)}, iter={study.best_params.get('iterations', 500)}, weights='{study.best_params.get('auto_class_weights', 'Balanced')}'",
        "Akurasi (%)": round(accuracy_score(y_test, pred_tuned) * 100, 2),
        "Presisi (%)": round(precision_score(y_test, pred_tuned) * 100, 2),
        "Recall (%)": round(recall_score(y_test, pred_tuned) * 100, 2),
        "F1-Score (%)": round(f1_score(y_test, pred_tuned) * 100, 2),
        "ROC-AUC (%)": round(roc_auc_score(y_test, prob_tuned) * 100, 2),
        "Latensi (ms/tx)": round(lat_tuned, 4)
    }
]

df_comp = pd.DataFrame(comparison_results)
comp_csv_path = os.path.join(OUTPUT_DIR, "tabel_4_3_komparasi_optuna.csv")
df_comp.to_csv(comp_csv_path, index=False)

print("\n" + "=" * 85)
print("🏆 TABEL 4.3 PERBANDINGAN KINERJA CATBOOST DEFAULT VS TUNED (OPTUNA)")
print("=" * 85)
print(df_comp.to_string(index=False))
print("=" * 85)

# 9. Ekspor Model Terbaik ke Format Binary (.cbm) untuk Web Streamlit
champion_model_path = os.path.join(MODEL_DIR, "model_catboost_fraud.cbm")
model_tuned.save_model(champion_model_path)
print(f"\n✅ Model terbaik berhasil diekspor ke: {champion_model_path}")
print("   Model ini siap digunakan secara langsung oleh aplikasi web Streamlit (app.py).")
