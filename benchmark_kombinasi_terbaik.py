# ==============================================================================
# SKRIPSI: BENCHMARK KOMBINASI TERBAIK CATBOOST
# Topik: Deteksi Fraud Transaksi Blockchain Ethereum
# Eksperimen:
# 1. 20 Fitur vs 19 Fitur (Dampak Penghapusan block_number)
# 2. Perbandingan Split Ratio (70:30, 80:20, 90:10)
# 3. Perbandingan Threshold Cohen's d (d >= 0.80, d >= 0.60, d >= 0.50)
# ==============================================================================

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "DeFiTransLyzer_combined_clean_v3.csv")
TARGET_COL = "flag"

# 1. Definisi Himpunan Fitur Berdasarkan Threshold Cohen's d
# A. 20 Fitur Awal (d >= 0.60 dengan block_number)
FEATURES_20 = [
    "nonce.nonce_coefficient_of_variation",
    "cumulativeGasUsed.cumulativeGasUsed_mode",
    "cumulativeGasUsed.cumulativeGasUsed_median",
    "erc_20_Symbol_End_Is_Digit",
    "erc_20_Symbol_Has_Digit",
    "cumulativeGasUsed.cumulativeGasUsed_average",
    "block_number",  # Fitur bias temporal
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

# B. 19 Fitur Rekomendasi (d >= 0.60 TANPA block_number - Pure Behavioral)
FEATURES_19 = [f for f in FEATURES_20 if f != "block_number"]

# C. 13 Fitur Ketat (d >= 0.80 murni perilaku tanpa block_number)
FEATURES_13_D80 = [
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
    "gas_efficiency"
]

# D. 22 Fitur Longgar (d >= 0.50 tanpa block_number)
FEATURES_22_D50 = FEATURES_19 + [
    "cumulativeGasUsed.cumulativeGasUsed_summation",
    "cumulativeGasUsed.cumulativeGasUsed_skewness",
    "number_of_unique_to_address"
]

ALL_REQUIRED_COLS = list(set(FEATURES_22_D50 + FEATURES_20 + [TARGET_COL]))

print("=" * 85)
print("🚀 MEMULAI EKSPERIMEN BENCHMARK KOMBINASI TERBAIK MODEL CATBOOST")
print("=" * 85)

# 2. Load Dataset Sekali Saja
print(f"\n[1/3] Membaca dataset dari: {DATASET_PATH}...")
start_load = time.time()
df = pd.read_csv(DATASET_PATH, usecols=ALL_REQUIRED_COLS, low_memory=False)
print(f"      Selesai membaca {len(df):,} baris data dalam {time.time() - start_load:.2f} detik.")

y = df[TARGET_COL].astype(int)

# 3. Matriks Skenario Pengujian
scenarios = [
    # --- Eksperimen A: Pengaruh Jumlah Fitur (Pada Split Standar 80:20) ---
    {
        "nama_skenario": "Skenario 1 (20 Fitur Awal - dengan block_number)",
        "fitur": FEATURES_20,
        "test_size": 0.20,
        "keterangan": "Threshold d >= 0.60 (Masih Mengandung Bias Temporal)"
    },
    {
        "nama_skenario": "Skenario 2 (19 Fitur - Drop block_number) [REKOMENDASI]",
        "fitur": FEATURES_19,
        "test_size": 0.20,
        "keterangan": "Threshold d >= 0.60 (100% Pure Behavioral Features)"
    },
    {
        "nama_skenario": "Skenario 3 (13 Fitur Ketat - d >= 0.80)",
        "fitur": FEATURES_13_D80,
        "test_size": 0.20,
        "keterangan": "Threshold d >= 0.80 (Kehilangan Fitur Topologi Akun)"
    },
    {
        "nama_skenario": "Skenario 4 (22 Fitur Longgar - d >= 0.50)",
        "fitur": FEATURES_22_D50,
        "test_size": 0.20,
        "keterangan": "Threshold d >= 0.50 (Memasukkan Fitur Gas Redundan)"
    },

    # --- Eksperimen B: Pengaruh Split Ratio (Pada 19 Fitur Murni) ---
    {
        "nama_skenario": "Skenario 5 (19 Fitur - Split 70:30)",
        "fitur": FEATURES_19,
        "test_size": 0.30,
        "keterangan": "Uji Pembagian Data 70% Train : 30% Test"
    },
    {
        "nama_skenario": "Skenario 6 (19 Fitur - Split 75:25)",
        "fitur": FEATURES_19,
        "test_size": 0.25,
        "keterangan": "Uji Pembagian Data 75% Train : 25% Test"
    },
    {
        "nama_skenario": "Skenario 7 (19 Fitur - Split 90:10)",
        "fitur": FEATURES_19,
        "test_size": 0.10,
        "keterangan": "Uji Pembagian Data 90% Train : 10% Test"
    }
]

benchmark_results = []

print(f"\n[2/3] Menjalankan {len(scenarios)} skenario pengujian komparatif...")

for idx, sc in enumerate(scenarios, 1):
    print("\n" + "-" * 85)
    print(f"▶ [{idx}/{len(scenarios)}] Menjalankan: {sc['nama_skenario']}")
    print(f"   • Jumlah Fitur : {len(sc['fitur'])} fitur")
    print(f"   • Pembagian    : {int((1-sc['test_size'])*100)}% Train / {int(sc['test_size']*100)}% Test")
    print(f"   • Deskripsi    : {sc['keterangan']}")
    
    # Preprocessing fitur spesifik
    X_sub = df[sc['fitur']].replace([np.inf, -np.inf], np.nan)
    X_sub_clean = X_sub.fillna(X_sub.median())
    
    # Train test split stratified
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub_clean, y, test_size=sc['test_size'], random_state=42, stratify=y
    )
    
    # Training CatBoost
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.08,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0
    )
    
    t_start = time.time()
    model.fit(X_tr, y_tr, eval_set=(X_te, y_te), early_stopping_rounds=30, verbose=False)
    t_train = time.time() - t_start
    
    # Evaluasi Prediksi
    t_infer_start = time.time()
    y_pred_prob = model.predict_proba(X_te)[:, 1]
    t_infer = time.time() - t_infer_start
    
    y_pred = (y_pred_prob >= 0.50).astype(int)
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_pred_prob)
    lat_ms = (t_infer / len(X_te)) * 1000
    
    res = {
        "Skenario": sc["nama_skenario"],
        "Jml_Fitur": len(sc["fitur"]),
        "Split_Ratio": f"{int((1-sc['test_size'])*100)}:{int(sc['test_size']*100)}",
        "Akurasi (%)": round(acc * 100, 2),
        "Presisi (%)": round(prec * 100, 2),
        "Recall (%)": round(rec * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "ROC-AUC (%)": round(auc * 100, 2),
        "Waktu_Latih (s)": round(t_train, 2),
        "Latensi (ms/tx)": round(lat_ms, 5),
        "Keterangan": sc["keterangan"]
    }
    benchmark_results.append(res)
    
    print(f"   => Hasil: Akurasi={res['Akurasi (%)']}%, Presisi={res['Presisi (%)']}%, Recall={res['Recall (%)']}%, F1={res['F1-Score (%)']}%, AUC={res['ROC-AUC (%)']}% (Waktu: {res['Waktu_Latih (s)']}s)")

# 4. Ringkasan Hasil dalam Bentuk Tabel
res_df = pd.DataFrame(benchmark_results)

# Simpan ke CSV untuk dimasukkan ke skripsi/Excel
output_csv_path = os.path.join(BASE_DIR, "hasil_benchmark_kombinasi.csv")
res_df.to_csv(output_csv_path, index=False)

print("\n" + "=" * 85)
print("🏆 HASIL AKHIR BENCHMARK KOMBINASI TERBAIK (Tersimpan di: hasil_benchmark_kombinasi.csv)")
print("=" * 85)
print(res_df[["Skenario", "Jml_Fitur", "Split_Ratio", "Akurasi (%)", "Presisi (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC (%)"]].to_string(index=False))
print("=" * 85)
