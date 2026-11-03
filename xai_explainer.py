# ==============================================================================
# XAI HUMAN-READABLE EXPLAINER MODULE
# Topik Skripsi: Deteksi Fraud Transaksi Blockchain Ethereum Menggunakan CatBoost
# Fungsi: Menerjemahkan luaran teknis SHAP & 20 fitur blockchain ke dalam:
#   1. Paragraf Narasi Penjelasan Eksekutif ("Bahasa Manusia")
#   2. Tabel Analisis Faktor & Pembanding Nilai Acuan Normal Blockchain
#   3. Diagram Visualisasi Komprehensif (Bar Kontribusi Risiko + Waterfall Plot)
# ==============================================================================

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import shap

def get_severity_from_proba(proba_pct):
    if proba_pct >= 85.0:
        return "Kritis"
    elif proba_pct >= 65.0:
        return "Tinggi"
    elif proba_pct >= 50.0:
        return "Sedang"
    else:
        return "Aman"

# ==============================================================================
# 1. KNOWLEDGE BASE 20 FITUR BLOCKCHAIN: BENCHMARK & MAKNA HUMANIS
# ==============================================================================

FEATURE_KB = {
    "erc_20_Symbol_End_Is_Digit": {
        "name": "Simbol ERC-20 Berakhiran Digit",
        "benchmark": "0 (Tidak ada digit di akhir simbol)",
        "unit": "digit",
        "format": lambda v: "1 (Ada Digit)" if v > 0.5 else "0 (Tidak Ada)",
        "meaning_fraud": "Simbol token berakhiran angka (contoh: USDT2, UNI8). Token resmi hampir tidak pernah menggunakan angka di akhir; ini adalah trik typosquatting untuk menipu korban dalam airdrop scam.",
        "meaning_legit": "Simbol token tidak berakhiran angka, mematuhi standar penamaan token resmi."
    },
    "erc_20_Symbol_Has_Digit": {
        "name": "Simbol ERC-20 Memiliki Digit",
        "benchmark": "0 (Tidak ada angka dalam simbol)",
        "unit": "digit",
        "format": lambda v: "1 (Ada Digit)" if v > 0.5 else "0 (Tidak Ada)",
        "meaning_fraud": "Simbol token disisipi angka numerik. Modus ini biasa digunakan untuk memalsukan nama token terkenal agar lolos pemeriksaan visual korban.",
        "meaning_legit": "Simbol token hanya terdiri dari karakter huruf alfabet standar tanpa sisipan angka tipuan."
    },
    "erc_20_Name_Has_Digit": {
        "name": "Nama Kontrak ERC-20 Memiliki Digit",
        "benchmark": "0 (Nama kontrak bersih tanpa angka)",
        "unit": "digit",
        "format": lambda v: "1 (Ada Digit)" if v > 0.5 else "0 (Tidak Ada)",
        "meaning_fraud": "Nama resmi kontrak token disisipi angka, indikasi kuat upaya cloning atau pemalsuan identitas proyek kripto.",
        "meaning_legit": "Nama resmi kontrak token teratur tanpa angka rekayasa."
    },
    "erc_721_Symbol_Lowercase_Count": {
        "name": "Huruf Kecil Simbol NFT ERC-721",
        "benchmark": "0 (Standar simbol kapital penuh)",
        "unit": "karakter",
        "format": lambda v: f"{int(v)} huruf kecil" if v > 0 else "0 (Huruf Besar)",
        "meaning_fraud": "Simbol NFT menggunakan huruf kecil tak lazim, sering ditemukan pada koleksi NFT phishing tiruan untuk menjebak kolektor.",
        "meaning_legit": "Simbol NFT mematuhi kaidah kapitalisasi standar (contoh: BAYC, PUNKS)."
    },
    "cumulativeGasUsed.cumulativeGasUsed_mode": {
        "name": "Modus Gas Kumulatif",
        "benchmark": "21,000 - 100,000 gas",
        "unit": "gas",
        "format": lambda v: f"{v:,.0f} gas",
        "meaning_fraud": "Modus konsumsi gas sangat masif, mencerminkan pemanggilan smart contract rumit secara berulang (multi-call / DeFi exploit).",
        "meaning_legit": "Konsumsi gas rutin berada dalam rentang efisien transaksi standar Ethereum."
    },
    "cumulativeGasUsed.cumulativeGasUsed_median": {
        "name": "Median Gas Kumulatif",
        "benchmark": "21,000 - 100,000 gas",
        "unit": "gas",
        "format": lambda v: f"{v:,.0f} gas",
        "meaning_fraud": "Nilai tengah gas transaksi sangat tinggi (>300.000 gas), lazim ditemui pada serangan flash loan atau arbitrase manipulatif.",
        "meaning_legit": "Nilai tengah gas wajar dan proporsional untuk interaksi dompet standar."
    },
    "cumulativeGasUsed.cumulativeGasUsed_average": {
        "name": "Rata-rata Gas Kumulatif",
        "benchmark": "21,000 - 120,000 gas",
        "unit": "gas",
        "format": lambda v: f"{v:,.0f} gas",
        "meaning_fraud": "Rata-rata beban komputasi transaksi sangat berat, pertanda interaksi kontrak bertingkat yang tidak wajar.",
        "meaning_legit": "Beban komputasi gas rata-rata rendah dan aman."
    },
    "cumulative_gas_used": {
        "name": "Total Gas Digunakan",
        "benchmark": "21,000 - 150,000 gas",
        "unit": "gas",
        "format": lambda v: f"{v:,.0f} gas",
        "meaning_fraud": "Akumulasi gas blok sangat besar (>1,500,000 gas), mencerminkan transaksi eksploitatif berskala besar dalam satu blok.",
        "meaning_legit": "Total gas yang dihabiskan tergolong normal untuk transfer atau swap sederhana."
    },
    "gas_efficiency": {
        "name": "Efisiensi Gas Transaksi",
        "benchmark": "0.80 - 1.20 (Optimal)",
        "unit": "rasio",
        "format": lambda v: f"{v:.2f}",
        "meaning_fraud": "Rasio efisiensi gas ekstrem (>20), menunjukkan skrip eksekusi batch contract otomatis yang agresif.",
        "meaning_legit": "Efisiensi penggunaan gas proporsional dengan alokasi gas limit yang wajar."
    },
    "cumulativeGasUsed.cumulativeGasUsed_standard_deviation": {
        "name": "Deviasi Standar Gas Kumulatif",
        "benchmark": "0 - 50,000 gas (Stabil)",
        "unit": "gas",
        "format": lambda v: f"{v:,.0f} gas",
        "meaning_fraud": "Variasi gas ekstrem menandakan fluktuasi liar eksekusi kontrak pintar yang berisiko manipulasi.",
        "meaning_legit": "Penggunaan gas stabil tanpa lonjakan deviasi tak terduga."
    },
    "cumulativeGasUsed.cumulativeGasUsed_coefficient_of_variation": {
        "name": "Variasi Gas Kumulatif (CV)",
        "benchmark": "0.00 - 0.50 (Konsisten)",
        "unit": "rasio",
        "format": lambda v: f"{v:.3f}",
        "meaning_fraud": "Tingkat ketidakpastian konsumsi gas tinggi mencerminkan perilaku transaksi yang tidak konsisten.",
        "meaning_legit": "Variasi gas rendah menunjukkan pola transaksi yang teratur dan wajar."
    },
    "number_of_unique_from_address": {
        "name": "Jumlah Alamat Pengirim Unik",
        "benchmark": "1 - 3 alamat dompet",
        "unit": "alamat",
        "format": lambda v: f"{int(v)} alamat",
        "meaning_fraud": "Terhubung dengan banyak dompet pengirim unik (pola fan-in tinggi), ciri khas dompet penampung hasil penipuan phishing/wallet drainer.",
        "meaning_legit": "Interaksi terpusat pada satu atau sedikit alamat wajar tanpa tanda pengumpulan aset massal."
    },
    "number_of_from_address": {
        "name": "Total Transaksi Pengirim (From)",
        "benchmark": "1 - 10 transaksi",
        "unit": "transaksi",
        "format": lambda v: f"{int(v)} transaksi",
        "meaning_fraud": "Frekuensi pengiriman keluar tinggi secara serentak, indikasi otomatisasi bot penguras dana.",
        "meaning_legit": "Volume transaksi keluar wajar untuk aktivitas pengguna manusia reguler."
    },
    "number_of_to_address": {
        "name": "Total Alamat Penerima (To)",
        "benchmark": "1 - 5 alamat",
        "unit": "alamat",
        "format": lambda v: f"{int(v)} alamat",
        "meaning_fraud": "Penyebaran transaksi ke banyak alamat tujuan dalam rentang singkat (pola airdrop spamming).",
        "meaning_legit": "Pengiriman ke alamat tujuan normal sesuai kebutuhan transfer pengguna."
    },
    "num_transaction": {
        "name": "Total Riwayat Transaksi Akun",
        "benchmark": "1 - 20 transaksi",
        "unit": "transaksi",
        "format": lambda v: f"{int(v)} transaksi",
        "meaning_fraud": "Lonjakan jumlah transaksi luar biasa tinggi mengindikasikan akun bot spammer atau jaringan drainer.",
        "meaning_legit": "Jumlah riwayat transaksi wajar mencerminkan aktivitas akun pengguna normal."
    },
    "nonce.nonce_coefficient_of_variation": {
        "name": "Variasi Nonce (CV)",
        "benchmark": "0.00 - 5.00 (Berurutan)",
        "unit": "rasio",
        "format": lambda v: f"{v:.2f}",
        "meaning_fraud": "Variasi urutan nonce tinggi (>8) membuktikan transaksi dikirim secara paralel atau diserobot oleh bot transaksi otomatis (MEV bot).",
        "meaning_legit": "Urutan nonce bertambah secara teratur dari waktu ke waktu sesuai perilaku transaksi dompet manusia."
    },
    "nonce.nonce_skewness": {
        "name": "Kemiringan Distribusi Nonce",
        "benchmark": "0.00 - 2.50 (Simetris)",
        "unit": "skor",
        "format": lambda v: f"{v:.2f}",
        "meaning_fraud": "Asimetri nonce ekstrem menandakan lonjakan transaksi tak terkendali dalam waktu amat singkat.",
        "meaning_legit": "Distribusi urutan transaksi teratur tanpa anomali percepatan."
    },
    "gas_prices.gas_prices_skewness": {
        "name": "Kemiringan Harga Gas",
        "benchmark": "0.00 - 1.50 (Harga Gas Stabil)",
        "unit": "skor",
        "format": lambda v: f"{v:.2f}",
        "meaning_fraud": "Kemiringan harga gas tinggi mencerminkan perang penawaran biaya gas (gas bidding war / front-running) untuk mendahului transaksi korban di mempool.",
        "meaning_legit": "Penawaran harga gas stabil mengikuti tarif standar jaringan Ethereum."
    },
    "block_number": {
        "name": "Nomor Blok Ethereum",
        "benchmark": "Blok Ethereum Terkini",
        "unit": "blok",
        "format": lambda v: f"#{int(v):,}",
        "meaning_fraud": "Ketinggian blok transaksi di blockchain Ethereum saat transaksi dieksekusi.",
        "meaning_legit": "Ketinggian blok transaksi di blockchain Ethereum saat transaksi dieksekusi."
    },
    "index": {
        "name": "Indeks Transaksi dalam Blok",
        "benchmark": "0 - 250 (Posisi dalam blok)",
        "unit": "posisi",
        "format": lambda v: f"Urutan #{int(v)}",
        "meaning_fraud": "Posisi transaksi dalam blok. Posisi awal sering diperebutkan bot arbitrase / sandwich attack untuk memanipulasi urutan transaksi.",
        "meaning_legit": "Urutan transaksi dalam blok terdistribusi wajar sesuai antrean reguler."
    }
}


# ==============================================================================
# 2. GENERATOR PARAGRAF NARASI DALAM BAHASA MANUSIA
# ==============================================================================

def generate_human_narrative(row_data, df_factors, is_fraud, confidence, severity=None):
    """
    Menghasilkan narasi terstruktur yang menjelaskan 'mengapa' transaksi ini
    dinyatakan FRAUD atau LEGITIMATE dalam bahasa Indonesia formal tanpa asteris berlebihan.
    """
    if severity is None:
        severity = get_severity_from_proba(confidence) if is_fraud else "Aman"

    # Pisahkan faktor pendorong risiko (+SHAP) dan penurun risiko (-SHAP)
    top_pos = df_factors[df_factors["SHAP"] > 0].sort_values(by="SHAP", ascending=False)
    top_neg = df_factors[df_factors["SHAP"] < 0].sort_values(by="SHAP", ascending=True)

    paragraphs = []

    # --- PARAGRAF 1: KESIMPULAN EKSEKUTIF ---
    if is_fraud:
        p1 = (
            f"Berdasarkan analisis model machine learning CatBoost terhadap 19 fitur on-chain Ethereum, "
            f"transaksi ini terdeteksi sebagai Fraud (Transaksi Berbahaya) dengan tingkat risiko sebesar "
            f"{confidence:.1f}% (Tingkat Bahaya: {severity}). Karakteristik teknis transaksi ini menyimpang "
            f"secara signifikan dari profil transaksi normal dan mengindikasikan adanya anomali on-chain berbahaya."
        )
    else:
        p1 = (
            f"Berdasarkan analisis model machine learning CatBoost terhadap 19 fitur on-chain Ethereum, "
            f"transaksi ini dinyatakan Legitimate (Transaksi Normal) dengan tingkat kepastian sebesar "
            f"{(100 - confidence):.1f}% (Probabilitas Risiko Fraud sangat rendah: {confidence:.1f}%, Tingkat Bahaya: Aman). "
            f"Pola transaksi sepenuhnya selaras dengan perilaku transaksi keuangan reguler pada jaringan Ethereum."
        )
    paragraphs.append(p1)

    # --- PARAGRAF 2: PENJELASAN AKAR MASALAH (THE 'WHY' IN HUMAN LANGUAGE) ---
    if is_fraud:
        reasons = []
        # Telusuri top 3 pemicu fraud
        for _, r in top_pos.head(3).iterrows():
            f_code = r["Fitur_Kode"]
            kb = FEATURE_KB.get(f_code, {})
            f_name = r["Fitur (Blockchain)"]
            f_val_str = r["Nilai_Format"]
            meaning = kb.get("meaning_fraud", "Nilai fitur menyimpang jauh dari batas transaksi normal.")
            reasons.append(f"Fitur {f_name} tercatat bernilai {f_val_str} (dampak risiko +{r['SHAP']:.4f}). {meaning}")

        reasons_text = " ".join(reasons)
        p2 = (
            f"Mengapa transaksi ini dinyatakan Fraud? Sistem Explainable AI (SHAP) mengungkap bahwa model "
            f"mendeteksi beberapa anomali on-chain kritis yang mendorong vonis kejahatan secara signifikan: "
            f"{reasons_text}"
        )
    else:
        # Transaksi Legitimate
        safe_factors = []
        for _, r in top_neg.head(3).iterrows():
            f_code = r["Fitur_Kode"]
            kb = FEATURE_KB.get(f_code, {})
            f_name = r["Fitur (Blockchain)"]
            f_val_str = r["Nilai_Format"]
            meaning = kb.get("meaning_legit", "Nilai fitur berada pada rentang yang wajar dan aman.")
            safe_factors.append(f"Fitur {f_name} berada pada angka {f_val_str} (meredam risiko sebesar {r['SHAP']:.4f}). {meaning}")

        safe_text = " ".join(safe_factors)
        p2 = (
            f"Mengapa transaksi ini dinyatakan Aman? Seluruh parameter teknis kunci berada dalam batas standar "
            f"ekosistem Ethereum yang sehat: {safe_text} Tidak ditemukan tanda-tanda manipulasi penamaan kontrak cerdas, "
            f"eksploitasi gas, ataupun aktivitas bot otomatis yang mencurigakan."
        )
    paragraphs.append(p2)

    # --- PARAGRAF 3: REKOMENDASI TINDAKAN PRAKTIS ---
    if is_fraud:
        p3 = (
            f"Rekomendasi Tindakan Keamanan: Tolak atau tunda transaksi ini. Pengguna dompet sangat disarankan "
            f"untuk tidak menandatangani persetujuan izin transfer (Approve / Permit) pada smart contract yang bersangkutan, "
            f"mencabut izin kontrak yang mencurigakan (melalui layanan seperti Revoke.cash), "
            f"dan memverifikasi ulang alamat penerima sebelum melanjutkan transaksi."
        )
    else:
        p3 = (
            f"Rekomendasi Tindakan: Transaksi ini aman untuk divalidasi dan diproses lebih lanjut. "
            f"Tidak ada langkah mitigasi khusus yang diperlukan karena tidak terdapat anomali perilaku on-chain yang membahayakan aset."
        )
    paragraphs.append(p3)

    return paragraphs


# ==============================================================================
# 3. GENERATOR TABEL FAKTOR BREAKDOWN & PEMBANDING BENCHMARK NORMAL
# ==============================================================================

def build_factor_comparison_table(row_data, shap_values, top_20_features):
    """
    Membangun DataFrame komparasi rapi berisi:
    1. Nama Fitur (Bahasa Indonesia)
    2. Nilai Transaksi Ini
    3. Nilai Acuan Normal (Blockchain)
    4. Kontribusi AI (SHAP Impact)
    5. Makna & Implikasi Risiko On-Chain
    """
    records = []
    
    for idx, f_code in enumerate(top_20_features):
        kb = FEATURE_KB.get(f_code, {})
        f_name = kb.get("name", f_code)
        bench = kb.get("benchmark", "Standar wajar")
        
        # Ambil nilai aktual
        val = row_data.get(f_code, 0.0)
        try:
            val_float = float(val)
            if math.isnan(val_float) or math.isinf(val_float):
                val_float = 0.0
        except (ValueError, TypeError):
            val_float = 0.0
            
        fmt_func = kb.get("format", lambda v: f"{v}")
        val_str = fmt_func(val_float)
        
        shap_val = float(shap_values[idx])
        
        if shap_val > 0:
            impact_badge = f"+{shap_val:.4f} (Pendorong Fraud)"
            meaning = kb.get("meaning_fraud", "Mendorong model ke arah vonis fraud.")
            arah = "Pendorong Fraud"
        else:
            impact_badge = f"{shap_val:.4f} (Peredam Risiko / Aman)"
            meaning = kb.get("meaning_legit", "Mendukung transaksi sebagai wajar/legitimate.")
            arah = "Mendukung Aman"
            
        records.append({
            "Fitur_Kode": f_code,
            "Fitur (Blockchain)": f_name,
            "Nilai Transaksi Ini": val_str,
            "Nilai Acuan Normal": bench,
            "Pengaruh AI (SHAP)": impact_badge,
            "Arah": arah,
            "Nilai_Raw": val_float,
            "Nilai_Format": val_str,
            "SHAP": shap_val,
            "Makna & Implikasi Risiko On-Chain": meaning
        })
        
    df_factors = pd.DataFrame(records)
    # Urutkan berdasarkan besaran absolut SHAP (faktor paling berpengaruh di posisi teratas)
    df_factors["Abs_SHAP"] = df_factors["SHAP"].abs()
    df_factors = df_factors.sort_values(by="Abs_SHAP", ascending=False).reset_index(drop=True)
    return df_factors


# ==============================================================================
# 4. GENERATOR DIAGRAM VISUALISASI (DUAL PLOT: BAR CHART & WATERFALL)
# ==============================================================================

def create_risk_contribution_bar_chart(df_factors, max_bars=8):
    """
    Membuat diagram batang horizontal yang memisahkan:
    - Merah: Faktor Pendorong Risiko Fraud (SHAP > 0)
    - Hijau/Biru: Faktor Penurun Risiko / Pendukung Aman (SHAP < 0)
    Dengan tampilan modern bergaya Apple.
    """
    # Ambil top pendorong fraud dan top pendukung aman
    top_pos = df_factors[df_factors["SHAP"] > 0].head(max_bars // 2)
    top_neg = df_factors[df_factors["SHAP"] < 0].head(max_bars // 2)
    
    plot_df = pd.concat([top_neg, top_pos]).sort_values(by="SHAP", ascending=True)
    
    if len(plot_df) == 0:
        plot_df = df_factors.head(max_bars).sort_values(by="SHAP", ascending=True)
        
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(plot_df) * 0.45)), dpi=150)
    
    y_pos = np.arange(len(plot_df))
    shap_vals = plot_df["SHAP"].values
    labels = plot_df["Fitur (Blockchain)"].values
    
    # Warna: Merah (#FF3B30) untuk penambah risiko, Hijau Apple (#34C759) untuk penurun risiko
    colors = ['#FF3B30' if val > 0 else '#34C759' for val in shap_vals]
    
    bars = ax.barh(y_pos, shap_vals, color=colors, height=0.62, alpha=0.9, edgecolor='none')
    
    ax.axvline(0, color='#8E8E93', linewidth=1.2, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight='500', color='#1D1D1F')
    ax.set_xlabel("Dampak SHAP terhadap Prediksi Model (Log-Odds)", fontsize=11, fontweight='600', color='#1D1D1F', labelpad=8)
    
    # Anotasi teks nilai di ujung bar
    for bar, val in zip(bars, shap_vals):
        width = bar.get_width()
        offset = 0.015 if width >= 0 else -0.015
        ha = 'left' if width >= 0 else 'right'
        color = '#D70015' if width >= 0 else '#248A3D'
        ax.text(width + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va='center', ha=ha, fontsize=9.5, fontweight='600', color=color)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D1D1D6')
    ax.spines['bottom'].set_color('#D1D1D6')
    ax.grid(axis='x', linestyle=':', alpha=0.4, color='#AEAEB2')
    
    # Tambahkan custom legend yang rapi
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF3B30', label='Pendorong Fraud (Menaikkan Risiko)'),
        Patch(facecolor='#34C759', label='Peredam Risiko (Mendukung Aman)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='#F5F5F7', edgecolor='none', fontsize=9.5)
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='#F5F5F7', edgecolor='none', fontsize=9.5)
    
    plt.tight_layout()
    return fig


# ==============================================================================
# 5. MASTER RENDERER UNTUK STREAMLIT UI
# ==============================================================================

def render_human_xai_dashboard(tx_identifier, row_data, input_df, explainer, top_20_features, is_fraud, confidence, severity=None):
    """
    Menampilkan dashboard XAI manusiawi yang lengkap pada UI Streamlit:
    1. Paragraf Naratif Penjelasan Eksekutif
    2. Tabel Perbandingan Faktor & Nilai Acuan Normal
    3. Diagram Visualisasi (Waterfall Plot + Diagram Batang Kontribusi)
    """
    if severity is None:
        severity = get_severity_from_proba(confidence) if is_fraud else "Aman"

    # 1. Pastikan input_df selalu bertipe numerik float murni dengan 20 fitur berurutan
    clean_dict = {}
    for f in top_20_features:
        try:
            v = float(row_data.get(f, 0.0))
            if math.isnan(v) or math.isinf(v):
                v = 0.0
        except (ValueError, TypeError):
            v = 0.0
        clean_dict[f] = v
    clean_input_df = pd.DataFrame([clean_dict])

    # 2. Hitung nilai SHAP
    shap_res = explainer(clean_input_df)
    shap_vals_arr = shap_res[0].values

    # 3. Bangun tabel faktor & analisis
    df_factors = build_factor_comparison_table(clean_dict, shap_vals_arr, top_20_features)

    # 4. Hasilkan narasi paragraf manusiawi
    narrative_paragraphs = generate_human_narrative(clean_dict, df_factors, is_fraud, confidence, severity=severity)

    # --------------------------------------------------------------------------
    # BAGIAN 1: PARAGRAF NARASI PENJELASAN DALAM BAHASA MANUSIA
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="apple-card" style="margin-top: 18px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div class="apple-card-header" style="margin-bottom: 0;">Penjelasan Hasil Analisis Transaksi (Explainable AI)</div>
            <span style="background: rgba(0, 113, 227, 0.1); color: #0071E3; padding: 4px 10px; border-radius: 20px; font-size: 11.5px; font-weight: 600;">
                CatBoost + SHAP
            </span>
        </div>
        <p style="font-size: 13.5px; color: #86868B; margin-bottom: 16px;">
            Berikut adalah dekonstruksi narasi dalam bahasa manusia yang menjelaskan transparansi alasan di balik vonis CatBoost AI.
        </p>
    """, unsafe_allow_html=True)

    # Render Paragraf 1 (Status Eksekutif)
    badge_style = "background: rgba(255, 59, 48, 0.1); color: #FF3B30; border: 1px solid rgba(255, 59, 48, 0.25);" if is_fraud else "background: rgba(52, 199, 89, 0.1); color: #34C759; border: 1px solid rgba(52, 199, 89, 0.25);"
    st.markdown(f"""
    <div style="border-radius: 12px; padding: 14px 18px; margin-bottom: 14px; {badge_style}">
        <p style="margin: 0; font-size: 14px; font-weight: 500; line-height: 1.6;">
            {narrative_paragraphs[0]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Render Paragraf 2 (Akar Masalah Teknis On-Chain)
    st.markdown(f"""
    <div style="background: #F9F9FB; border-radius: 12px; padding: 14px 18px; margin-bottom: 14px; border: 1px solid #E5E5EA;">
        <p style="margin: 0; font-size: 13.5px; color: #1D1D1F; line-height: 1.65;">
            {narrative_paragraphs[1]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Render Paragraf 3 (Rekomendasi Tindakan)
    rec_box = "background: rgba(255, 149, 0, 0.08); border: 1px solid rgba(255, 149, 0, 0.25);" if is_fraud else "background: rgba(0, 113, 227, 0.06); border: 1px solid rgba(0, 113, 227, 0.18);"
    st.markdown(f"""
    <div style="border-radius: 12px; padding: 12px 18px; margin-bottom: 8px; {rec_box}">
        <p style="margin: 0; font-size: 13px; color: #1D1D1F; line-height: 1.55;">
            {narrative_paragraphs[2]}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # BAGIAN 2: TABEL ANALISIS FAKTOR & PEMBANDING NILAI ACUAN NORMAL
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="apple-card" style="margin-top: 18px;">
        <div class="apple-card-header">Tabel Faktor Risiko Transaksi vs Nilai Acuan Normal Blockchain</div>
        <div style="font-size: 13.5px; color: #86868B; margin-bottom: 14px;">
            Tabel berikut membandingkan secara langsung parameter transaksi yang diinspeksi terhadap tolok ukur (benchmark) transaksi sehat pada jaringan Ethereum beserta kontribusi matematis SHAP.
        </div>
    """, unsafe_allow_html=True)

    # Tampilkan tabel interaktif
    display_table = df_factors[["Fitur (Blockchain)", "Nilai Transaksi Ini", "Nilai Acuan Normal", "Pengaruh AI (SHAP)", "Makna & Implikasi Risiko On-Chain"]]
    
    st.dataframe(
        display_table,
        use_container_width=True,
        height=360,
        column_config={
            "Fitur (Blockchain)": st.column_config.TextColumn("Fitur (Blockchain)", width="medium"),
            "Nilai Transaksi Ini": st.column_config.TextColumn("Nilai Transaksi Ini", width="small"),
            "Nilai Acuan Normal": st.column_config.TextColumn("Nilai Acuan Normal", width="medium"),
            "Pengaruh AI (SHAP)": st.column_config.TextColumn("Pengaruh AI (SHAP)", width="medium"),
            "Makna & Implikasi Risiko On-Chain": st.column_config.TextColumn("Makna & Implikasi Risiko On-Chain", width="large")
        }
    )
    
    st.caption("Keterangan: Nilai positif menandakan kontribusi pendorong ke arah fraud, sedangkan nilai negatif menandakan faktor peredam risiko yang mendukung transaksi normal.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # BAGIAN 3: DIAGRAM VISUALISASI DUA PERSPEKTIF
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="apple-card" style="margin-top: 18px;">
        <div class="apple-card-header">Diagram Visualisasi Kontribusi Fitur (XAI Decomposition)</div>
        <div style="font-size: 13.5px; color: #86868B; margin-bottom: 16px;">
            Visualisasi grafis kontribusi fitur dalam menggeser probabilitas prediksi CatBoost.
        </div>
    """, unsafe_allow_html=True)

    tab_diag1, tab_diag2 = st.tabs(["Diagram Kontribusi Faktor Risiko", "Grafik SHAP Waterfall Decomposition"])

    with tab_diag1:
        st.markdown("**Diagram Kontribusi Faktor Risiko Terbesar:**")
        fig_bar = create_risk_contribution_bar_chart(df_factors, max_bars=10)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with tab_diag2:
        st.markdown("**Dekomposisi Nilai Dasar Model Menuju Prediksi Akhir (Waterfall):**")
        fig_wf, ax_wf = plt.subplots(figsize=(9.5, 4.8), dpi=150)
        shap.plots.waterfall(shap_res[0], show=False)
        plt.tight_layout()
        st.pyplot(fig_wf)
        plt.close(fig_wf)

    st.markdown("</div>", unsafe_allow_html=True)
    
    return df_factors, narrative_paragraphs
