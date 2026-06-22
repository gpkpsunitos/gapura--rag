# Panduan Instalasi Gapura RAG

Sistem Retrieval-Augmented Generation (RAG) ini dioptimalkan untuk performa tinggi menggunakan OpenRouter (LLM) dan Pinecone (Vector Store).

## 1. Persiapan Lingkungan

Pastikan Anda telah menginstal **Python 3.10** atau lebih baru.

### 1.1 Persiapan Virtual Environment (Direkomendasikan)

#### Untuk Windows (PowerShell/CMD):
1. Buka PowerShell atau Command Prompt di folder proyek ini.
2. Jalankan perintah:
   ```powershell
   python -m venv venv
   ```
3. Aktifkan virtual environment:
   ```powershell
   .\venv\Scripts\activate
   ```
   *(Catatan: Jika muncul error kebijakan eksekusi di PowerShell, jalankan `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` terlebih dahulu)*

#### Untuk Mac/Linux (Terminal):
1. Buka Terminal di folder proyek ini.
2. Jalankan perintah:
   ```bash
   python3 -m venv venv
   ```
3. Aktifkan virtual environment:
   ```bash
   source venv/bin/activate
   ```

### 1.2 Instalasi Dependensi
Setelah virtual environment aktif (muncul tanda `(venv)` di sebelah kiri prompt), jalankan:
```bash
pip install -r requirements.txt
```

## 2. Konfigurasi API (.env)

Buat file bernama `.env` di folder utama (jika belum ada) dan isi dengan kunci API berikut:

```env
HF_TOKEN=token_huggingface_anda
PINECONE_API_KEY=kunci_pinecone_anda
PINECONE_INDEX=gapura-rag-v2-1024d
OPENROUTER_API_KEY=kunci_openrouter_anda
```

## 3. Menjalankan Aplikasi

### Menjalankan Server Lokal
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Buka [http://localhost:7860](http://localhost:7860) di browser Anda.

## 4. Penggunaan

1.  **Upload PDF**: Gunakan tab upload untuk memasukkan dokumen PDF (mendukung 200+ halaman).
2.  **Tanya Jawab**: Masukkan pertanyaan Anda. Sistem akan mencari referensi yang relevan secara paralel dan memberikan jawaban yang didukung oleh sumber dokumen.
3.  **Streaming**: Jawaban akan muncul secara real-time (syllable-by-syllable).

## 5. Deployment ke Hugging Face Spaces

1.  Push kode ke repository Hugging Face.
2.  Setel `OPENROUTER_API_KEY`, `PINECONE_API_KEY`, and `HF_TOKEN` di menu **Settings -> Variables and Secrets** pada Space Anda.

---
[ARCHITECT'S AUDIT]
- Optimasi: Pencarian paralel, Streaming token, Pengurangan Candidate Multiplier (x10).
- Mesin: Llama 3.3 70B Instruct (via OpenRouter).
