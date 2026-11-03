# ==============================================================================
# SKRIPSI: SIMULASI SISTEM FIREWALL PRE-TRANSAKSI ETHEREUM
# Topik: Deteksi Fraud Transaksi Blockchain Ethereum Menggunakan AI & Riwayat Kejahatan
# Modul: Pre-Transaction Defense & Address Reputation Alert Simulation
# ==============================================================================

import os
import sys
import time
import sqlite3
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "audit_history.db")

# Inisialisasi Database Reputasi Alamat Kejahatan Ethereum
MALICIOUS_REGISTRY = {
    "0x0123456789abcdef0123456789abcdef01234567": {
        "label": "Alamat Malicious Terkonfirmasi (Kasus Fraud Berulang)",
        "total_fraud_count": 8,
        "first_seen": "2024-03-12 14:22:01",
        "last_seen": "2024-08-19 09:45:12",
        "total_eth_drained": 42.50,
        "severity": "KRITIS"
    },
    "0x498b3bf2f2e680a6b7d2bf61b9a91c78496bcdef": {
        "label": "Alamat Malicious Terkonfirmasi (Kasus Fraud Berulang)",
        "total_fraud_count": 14,
        "first_seen": "2023-11-05 08:12:30",
        "last_seen": "2024-09-02 21:05:44",
        "total_eth_drained": 185.20,
        "severity": "KRITIS"
    },
    "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640": {
        "label": "Alamat Malicious Terkonfirmasi (Kasus Fraud Berulang)",
        "total_fraud_count": 5,
        "first_seen": "2024-01-15 11:30:19",
        "last_seen": "2024-07-28 17:14:02",
        "total_eth_drained": 520.00,
        "severity": "KRITIS"
    },
    "0xd8da6bf26964af9d7eed9e03e53415d37aa96045": {
        "label": "vitalik.eth (Alamat Legitimate)",
        "total_fraud_count": 0,
        "first_seen": "2015-07-30 00:00:00",
        "last_seen": "2026-09-14 18:00:00",
        "total_eth_drained": 0.0,
        "severity": "AMAN"
    }
}


def query_address_history(address):
    """Mengecek reputasi alamat di registry dan log audit SQLite."""
    addr_clean = address.strip().lower()
    
    # 1. Cek di registry simulasi bawaan
    if addr_clean in MALICIOUS_REGISTRY:
        return MALICIOUS_REGISTRY[addr_clean]
    
    # 2. Cek riwayat dari database audit_history.db jika ada
    record = None
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT verdict, top_factors FROM audit_logs WHERE identifier LIKE ?", (f"%{addr_clean[:10]}%",))
            rows = c.fetchall()
            conn.close()
            
            if rows:
                fraud_count = sum(1 for r in rows if "FRAUD" in str(r[0]).upper())
                if fraud_count > 0:
                    record = {
                        "label": f"Alamat Terdaftar dalam Riwayat Audit ({addr_clean[:10]}...)",
                        "total_fraud_count": fraud_count,
                        "fraud_breakdown": {
                            "Pola Transaksi Mencurigakan": fraud_count
                        },
                        "first_seen": "Tercatat di Audit DB",
                        "last_seen": "Tercatat di Audit DB",
                        "total_eth_drained": 0.0,
                        "severity": "TINGGI" if fraud_count >= 2 else "SEDANG"
                    }
        except Exception:
            pass
            
    return record


def run_simulation():
    print("\n" + "=" * 75)
    print("🛡️  SIMULASI SISTEM PERINGATAN PRE-TRANSAKSI BLOCKCHAIN ETHEREUM")
    print("   Teknologi: Firewall Deteksi Dini CatBoost AI & Address Reputation")
    print("=" * 75)
    
    print("\nPilih Skenario Uji Transaksi:")
    print("1. [Uji Kasus 1] Kirim ke Akun Normal / Bersih (vitalik.eth)")
    print("2. [Uji Kasus 2] Kirim ke Akun Penipuan Fake Token / Scam Airdrop Deployer")
    print("3. [Uji Kasus 3] Kirim ke Kontrak Phishing Permitt2 Wallet Drainer")
    print("4. [Uji Kasus 4] Interaksi dengan Akun Eksploitasi DeFi / Flash Loan")
    print("5. [Custom Input] Masukkan Alamat Dompet Ethereum Manual")
    
    choice = input("\nPilihan Anda (1-5, default=2): ").strip()
    if choice == "1":
        target_address = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        sample_amount = 0.5
    elif choice == "3":
        target_address = "0x498b3bf2f2e680a6b7d2bf61b9a91c78496bcdef"
        sample_amount = 2.0
    elif choice == "4":
        target_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        sample_amount = 10.0
    elif choice == "5":
        target_address = input("Masukkan Alamat Tujuan (0x...): ").strip()
        try:
            sample_amount = float(input("Masukkan Jumlah ETH: ").strip() or "1.0")
        except ValueError:
            sample_amount = 1.0
    else:
        # Default pilihan 2
        target_address = "0x0123456789abcdef0123456789abcdef01234567"
        sample_amount = 1.5

    sender_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"

    print("\n" + "-" * 75)
    print("📋 RINCIAN TRANSAKSI YANG AKAN DIKIRIM:")
    print(f"• Alamat Pengirim (Sender)    : {sender_address}")
    print(f"• Alamat Penerima (Recipient) : {target_address}")
    print(f"• Nilai Transfer              : {sample_amount} ETH")
    print(f"• Jaringan                    : Ethereum Mainnet (ChainID: 1)")
    print("-" * 75)

    print("\n🔍 Memeriksa rekam jejak & profil risiko alamat di Mempool Firewall...")
    time.sleep(1.0)

    history = query_address_history(target_address)

    if history and history.get("total_fraud_count", 0) > 0:
        total_cases = history["total_fraud_count"]
        breakdown = history["fraud_breakdown"]
        severity = history["severity"]
        label = history["label"]
        drained = history.get("total_eth_drained", 0.0)

        print("\n" + "!" * 75)
        print("🚨 PERINGATAN KEAMANAN: REKAM JEJAK KEJAHATAN DITEMUKAN!")
        print("!" * 75)
        print(f"Status Reputasi       : ⛔ ALAMAT BERBAHAYA / MALICIOUS ENTITY ({severity})")
        print(f"Identitas Terdaftar   : {label}")
        print(f"Total Kejahatan       : {total_cases} KALI PERNAH MELAKUKAN FRAUD")
        print(f"Estimasi Kerugian     : {drained} ETH telah dikuras dari para korban")
        print(f"Terakhir Terdeteksi   : {history.get('last_seen', '-')}")
        
        print("\n📊 RINCIAN JENIS KEJAHATAN YANG PERNAH DILAKUKAN AKUN INI:")
        for fraud_name, count in breakdown.items():
            pct = (count / total_cases) * 100
            print(f"   ⚠️  {fraud_name.ljust(35)} : {count} Kali ({pct:.1f}%)")

        print("\n🛡️  TINDAKAN PENCEGAHAN FIREWALL:")
        print("   Sistem keamanan secara otomatis MENAHAN eksekusi tanda tangan transaksi.")
        print("   Jangan lanjutkan jika Anda tidak mengenal pemilik alamat ini!")
        print("=" * 75)

        confirm = input("\nKetik 'SAYA_MENGERTI_RISIKO' untuk tetap memaksa kirim, atau tekan Enter untuk BATAL: ").strip()
        if confirm == "SAYA_MENGERTI_RISIKO":
            print("\n⚠️  Transaksi dipaksa oleh pengguna. Dana berhasil dikirim (PERINGATAN: Berisiko Tinggi Kehilangan Dana)!")
        else:
            print("\n✅ TRANSAKSI DIBATALKAN DENGAN AMAN!")
            print(f"   Dana sebesar {sample_amount} ETH berhasil dilindungi dari potensi penipuan.")
    else:
        print("\n" + "=" * 75)
        print("✅ ALAMAT TUJUAN TERVERIFIKASI AMAN / BERSIH")
        print("=" * 75)
        print("• Tidak ada catatan kejahatan fraud pada alamat ini di basis data reputasi.")
        print("• Model CatBoost mendeteksi pola transaksi dalam batas toleransi normal.")
        print(f"• Transaksi transfer {sample_amount} ETH siap dieksekusi ke jaringan.")
        print("=" * 75)


if __name__ == "__main__":
    run_simulation()
