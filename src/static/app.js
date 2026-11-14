/* Mini-AutoML front-end: vanilla JS, hash routing, talks to the FastAPI backend in server.py.
 *
 * Routes
 *   #/new            upload a dataset, profile its columns, pick a target, train
 *   #/runs           experiment history (outputs/experiments_log.csv)
 *   #/runs/<run_id>  one run: scores, evaluation plots, SHAP, threshold curve, files
 *
 * UI text lives in I18N (English + Bahasa Indonesia). Messages that come from the
 * server or the pipeline logs stay in English.
 */
"use strict";

/* ------------------------------------------------------------------ i18n */

const I18N = {
  en: {
    "doc.title": "Mini-AutoML",
    "skip": "Skip to content",
    "nav.label": "Main",
    "nav.new": "New run",
    "nav.runs": "Runs",
    "theme.label": "Dark theme",
    "theme.toDark": "Switch to dark theme",
    "theme.toLight": "Switch to light theme",
    "lang.label": "Language",
    "footer.text": "Binary classification with six models and a file-based experiment log in {code}.",
    "footer.api": "API reference",
    "loading": "Loading…",
    "retry": "Try again",
    "error.network": "Cannot reach the server. Check that the container is running.",
    "notFound.title": "Page not found",
    "notFound.body": "This address does not match any page in Mini-AutoML.",
    "notFound.cta": "Go to New run",

    "new.loading": "Loading dataset…",
    "new.loadError": "Could not load the current dataset.",
    "upload.title": "Start a new run",
    "upload.lede": "Upload a table with one column you want to predict. That column needs exactly two classes, for example Yes/No, 1/0 or M/B.",
    "upload.keep": "Keep current dataset",
    "upload.drop": "Choose a CSV file or drop it here",
    "upload.hint": "Comma, semicolon, tab and pipe separators are detected automatically. Accepted: .csv, .tsv, .txt",
    "upload.badType": "{name} is not a .csv, .tsv or .txt file.",
    "upload.progress": "Uploading and profiling {name} ({size})…",

    "col.numeric": "Numeric",
    "col.text": "Text",
    "col.distinct": "{n} distinct",
    "col.canBeTarget": "can be the target",
    "col.notBinary": "not binary",
    "col.idNote": "Every value is unique: dropped automatically as an ID",
    "col.missing": "{n} missing",
    "col.selected": "Selected as target",
    "col.mean": "mean {v}",
    "col.other.one": "{n} other value",
    "col.other.many": "{n} other values",

    "ds.crumb": "Dataset",
    "ds.rows": "rows",
    "ds.columns": "columns",
    "ds.separated": "{sep} separated, {enc}",
    "ds.uploaded": "uploaded {when}",
    "ds.replace": "Replace dataset",
    "ds.heading": "Choose the column to predict",
    "ds.binaryCount": "{n} of {total} columns have exactly two values and can be used as the target.",
    "ds.noBinary": "No column has exactly two values, so this dataset cannot be used for binary classification. Upload a different file.",
    "ds.onlyBinary": "Only show columns with two values",
    "ds.legend": "Target column",
    "ds.selectHint": "Select a target column to enable training.",
    "ds.target": "Target",
    "ds.starting": "Starting the pipeline…",
    "sep.,": "comma", "sep.;": "semicolon", "sep.\t": "tab", "sep.|": "pipe", "sep.other": "custom",

    "step.preprocess": "Validate and preprocess the data",
    "step.train": "Train {model}, tune the threshold, explain the model",
    "step.n": "Step {n}",
    "job.running": "Running · {t}",
    "job.done": "Done · {t}",
    "job.failed": "Failed",
    "job.skipped": "Skipped",
    "job.waiting": "Waiting",
    "job.failedTitle": "Training failed.",
    "job.chooseAnother": "Choose another target",
    "job.seeRuns": "See previous runs",
    "job.titleRunning": "Training: {target}",
    "job.titleStopped": "Run stopped: {target}",
    "job.titleDone": "Model trained: {target}",
    "job.started": "Started {time}",
    "job.elapsed": "Elapsed {t}",
    "job.stoppedAfter": "Stopped after {t}",
    "job.showLog": "Show log",
    "job.lost": "Lost track of the training job.",
    "anim.preprocess": "Scanning rows, cleaning columns and encoding features",
    "anim.train": "Training {model} and scanning thresholds for the best F1",
    "anim.trainOf": "Training {model} ({i} of {n}) and scanning thresholds for the best F1",
    "anim.done": "Done. Opening the results…",
    "anim.pause": "Pause animation",
    "anim.play": "Play animation",
    "time.s": "{s} s",
    "time.ms": "{m} min {s} s",

    "runs.title": "Runs",
    "runs.loading": "Loading runs…",
    "runs.loadError": "Could not load the experiment log.",
    "runs.emptyTitle": "No runs yet",
    "runs.emptyBody": "Every training run is recorded here with its scores, plots and model file.",
    "runs.emptyCta": "Upload a dataset",
    "runs.lede.one": "1 run recorded in {code}. Scores are measured on each dataset's 20% hold-out split, so compare runs of the same target only.",
    "runs.lede.many": "{n} runs recorded in {code}. Scores are measured on each dataset's 20% hold-out split, so compare runs of the same target only.",
    "runs.download": "Download log (CSV)",
    "runs.caption": "Training runs, sortable by column",
    "runs.positive": "positive: {v}",

    "m.run": "Run", "m.target": "Target", "m.f1": "F1", "m.f1score": "F1 score", "m.precision": "Precision",
    "m.recall": "Recall", "m.prauc": "PR-AUC", "m.accuracy": "Accuracy", "m.threshold": "Threshold",

    "run.loading": "Loading run…",
    "run.loadError": "Could not load this run.",
    "run.notFoundTitle": "Run not found",
    "run.notFoundBody": "{id} is not in the experiment log. It may have been deleted.",
    "run.back": "Back to all runs",
    "run.predicting": "Predicting {target}",
    "run.positiveClass": "positive class {pos}",
    "run.vs": "vs {neg}",
    "run.evaluatedOn": "evaluated on {n} hold-out rows",
    "run.trainAnother": "Train another model",
    "run.tabs": "Run details",
    "tab.evaluation": "Evaluation", "tab.explain": "Explainability", "tab.threshold": "Threshold", "tab.files": "Files",
    "fig.missing": "This plot was not generated for this run.",
    "fig.cm": "<strong>Confusion matrix</strong> at the chosen threshold of {t}: correct predictions on the diagonal, errors off it.",
    "fig.roc": "<strong>ROC curve.</strong> True positive rate against false positive rate across all thresholds. The red dot is the chosen operating point.",
    "fig.pr": "<strong>Precision-recall curve.</strong> More informative than ROC when the positive class is rare. The dashed line is a random classifier.",
    "fig.shap": "<strong>SHAP summary</strong> for a sample of the hold-out rows.",
    "eval.heading": "What these numbers mean for {pos}",
    "eval.p1": "At the threshold of <strong>{t}</strong>, {p} of the rows the model labels <strong>{pos}</strong> really are {pos} (precision).",
    "eval.p2": "Of all rows that really are {pos}, the model finds {r} (recall).",
    "eval.p3": "PR-AUC of <strong>{v}</strong> summarises the whole curve, independent of the threshold.",
    "shap.heading": "How to read this plot",
    "shap.p1": "Each row is a feature, ordered by its overall influence on the model. Each dot is one row from the hold-out set.",
    "shap.p2": "A dot to the right pushed the prediction towards <strong>{pos}</strong>; a dot to the left pushed it away. The colour shows the feature's value for that row: red is high, blue is low.",
    "shap.p3": "If a single feature dominates and the scores look too good, check whether it leaks the answer (for example a column recorded after the outcome happened).",
    "th.heading": "Why {t} and not 0.50",
    "th.p1": "The model outputs a probability. A row is labelled <strong>{pos}</strong> when that probability is at least the threshold.",
    "th.p2": "The pipeline tried every threshold from 0.10 to 0.90 in steps of 0.01 on the hold-out set and kept the one with the highest F1 score.",
    "th.aria": "F1 score by decision threshold. Highest F1 {f} at threshold {t}.",
    "th.axis": "Decision threshold",
    "th.marker": "F1 {f} at {t}",
    "th.none": "No threshold sweep was recorded for this run.",
    "files.heading": "Files",
    "files.params": "Parameters",
    "files.model": "Trained model. Load it with xgboost.XGBClassifier().load_model().",
    "files.metrics": "All metrics, class labels and the full threshold sweep for this run.",
    "files.download": "Download",
    "files.na": "Not available for this run.",
    "files.trainRows": "Training rows",
    "files.features": "Features after encoding",
    "pos.fallback": "the positive class",
    "pos.short": "positive",

    "ds.modelHeading": "Choose the model",
    "ds.modelLede": "Every model gets the same split, the same class-imbalance handling and its own F1-tuned threshold, so the scores are comparable.",
    "ds.trainOne": "Train {model} model",
    "ds.trainAll": "Train and compare {n} models",
    "ds.modelLabel": "Model",
    "ds.allModels": "all {n} models",
    "ds.excludeHeading": "Exclude columns (optional)",
    "ds.excludeLede": "Leave out columns the model must not learn from, for example values that are only known after the outcome, such as the duration of a sales call. ID-like and constant columns are already removed automatically.",
    "ds.excludeToggle": "Choose columns to exclude",
    "ds.excludeCount.none": "none selected",
    "ds.excludeCount": "{n} selected",
    "ds.excludedLabel": "Excluded",
    "ds.excludeTarget": "target",
    "job.excluded": "excluded: {cols}",
    "files.excluded": "Excluded columns",
    "files.none": "none",
    "model.compare": "Compare all models",
    "model.compare.desc": "Trains all {n} models one after another on the same split and shows them side by side.",
    "model.xgboost.desc": "Gradient-boosted trees. A strong default for tabular data.",
    "model.random_forest.desc": "Many independent decision trees, averaged. Stable and less prone to overfitting.",
    "model.logistic_regression.desc": "Linear baseline. Shows whether a complex model is needed at all.",
    "model.hist_gradient_boosting.desc": "scikit-learn's histogram-based gradient boosting. SHAP is estimated by sampling.",
    "model.lightgbm.desc": "Leaf-wise gradient boosting. Fast on large datasets.",
    "model.catboost.desc": "Gradient boosting with ordered boosting. Often strong without tuning.",
    "m.model": "Model",
    "job.viewFinished.one": "Open the finished run",
    "job.viewFinished.many": "Compare the {n} finished runs",
    "runs.select": "Select",
    "runs.selectRun": "Select {id} for comparison",
    "runs.compare": "Compare selected ({n})",
    "runs.compareHint": "Tick two or more runs to compare them side by side.",
    "run.modelMeta": "model {model}",
    "cmp.title": "Model comparison",
    "cmp.loading": "Loading comparison…",
    "cmp.heading": "Comparing {n} models on {target}",
    "cmp.headingMixed": "Comparing {n} runs",
    "cmp.lede": "Sorted by F1 score. Runs trained in the same job share the same 80/20 split, so their scores are directly comparable.",
    "cmp.mixed": "These runs predict different targets, so their scores are not comparable with each other.",
    "cmp.missing.one": "1 run could not be loaded and is left out.",
    "cmp.missing.many": "{n} runs could not be loaded and are left out.",
    "cmp.notEnough": "Pick at least two runs to compare.",
    "cmp.bestF1": "best F1",
    "cmp.bestPr": "best PR-AUC",
    "cmp.chart": "F1 score by model",
    "cmp.caption": "Model comparison, sorted by F1 score",
    "files.modelPrefix": "Trained {model} model.",
    "files.load.json": "Load it with xgboost.XGBClassifier().load_model().",
    "files.load.joblib": "Load it with joblib.load(). This format uses pickle: only open files you trust.",
    "files.load.txt": "Load it with lightgbm.Booster(model_file=...).",
    "files.load.cbm": "Load it with catboost.CatBoostClassifier().load_model().",
    "files.classWeight": "class_weight",
    "files.balanced": "balanced",
    "shap.note.hist_gradient_boosting": "For HistGradientBoosting, SHAP values are estimated by sampling (up to 300 rows), so they are approximate.",
    "shap.note.logistic_regression": "For logistic regression, SHAP values are computed on standardised features; the colours still show the original values.",
  },
  id: {
    "doc.title": "Mini-AutoML",
    "skip": "Langsung ke konten",
    "nav.label": "Utama",
    "nav.new": "Run baru",
    "nav.runs": "Riwayat",
    "theme.label": "Tema gelap",
    "theme.toDark": "Ganti ke tema gelap",
    "theme.toLight": "Ganti ke tema terang",
    "lang.label": "Bahasa",
    "footer.text": "Klasifikasi biner dengan enam model dan log eksperimen berbasis file di {code}.",
    "footer.api": "Referensi API",
    "loading": "Memuat…",
    "retry": "Coba lagi",
    "error.network": "Server tidak bisa dihubungi. Pastikan kontainer sedang berjalan.",
    "notFound.title": "Halaman tidak ditemukan",
    "notFound.body": "Alamat ini tidak cocok dengan halaman mana pun di Mini-AutoML.",
    "notFound.cta": "Ke Run baru",

    "new.loading": "Memuat dataset…",
    "new.loadError": "Dataset saat ini tidak bisa dimuat.",
    "upload.title": "Mulai run baru",
    "upload.lede": "Unggah tabel yang berisi satu kolom yang ingin diprediksi. Kolom itu harus punya tepat dua kelas, misalnya Yes/No, 1/0 atau M/B.",
    "upload.keep": "Pakai dataset saat ini",
    "upload.drop": "Pilih file CSV atau seret ke sini",
    "upload.hint": "Pemisah koma, titik koma, tab dan pipe dideteksi otomatis. Format: .csv, .tsv, .txt",
    "upload.badType": "{name} bukan file .csv, .tsv atau .txt.",
    "upload.progress": "Mengunggah dan memprofilkan {name} ({size})…",

    "col.numeric": "Numerik",
    "col.text": "Teks",
    "col.distinct": "{n} nilai unik",
    "col.canBeTarget": "bisa jadi target",
    "col.notBinary": "bukan biner",
    "col.idNote": "Semua nilai unik: otomatis dibuang sebagai ID",
    "col.missing": "{n} kosong",
    "col.selected": "Dipilih sebagai target",
    "col.mean": "rata-rata {v}",
    "col.other.one": "{n} nilai lain",
    "col.other.many": "{n} nilai lain",

    "ds.crumb": "Dataset",
    "ds.rows": "baris",
    "ds.columns": "kolom",
    "ds.separated": "pemisah {sep}, {enc}",
    "ds.uploaded": "diunggah {when}",
    "ds.replace": "Ganti dataset",
    "ds.heading": "Pilih kolom yang ingin diprediksi",
    "ds.binaryCount": "{n} dari {total} kolom punya tepat dua nilai dan bisa dipakai sebagai target.",
    "ds.noBinary": "Tidak ada kolom dengan tepat dua nilai, jadi dataset ini tidak bisa dipakai untuk klasifikasi biner. Unggah file lain.",
    "ds.onlyBinary": "Tampilkan hanya kolom dengan dua nilai",
    "ds.legend": "Kolom target",
    "ds.selectHint": "Pilih kolom target untuk mulai melatih.",
    "ds.target": "Target",
    "ds.starting": "Menjalankan pipeline…",
    "sep.,": "koma", "sep.;": "titik koma", "sep.\t": "tab", "sep.|": "pipe", "sep.other": "kustom",

    "step.preprocess": "Validasi dan praproses data",
    "step.train": "Latih {model}, atur threshold, jelaskan model",
    "step.n": "Langkah {n}",
    "job.running": "Berjalan · {t}",
    "job.done": "Selesai · {t}",
    "job.failed": "Gagal",
    "job.skipped": "Dilewati",
    "job.waiting": "Menunggu",
    "job.failedTitle": "Training gagal.",
    "job.chooseAnother": "Pilih target lain",
    "job.seeRuns": "Lihat run sebelumnya",
    "job.titleRunning": "Training: {target}",
    "job.titleStopped": "Run berhenti: {target}",
    "job.titleDone": "Model selesai dilatih: {target}",
    "job.started": "Mulai {time}",
    "job.elapsed": "Sudah {t}",
    "job.stoppedAfter": "Berhenti setelah {t}",
    "job.showLog": "Lihat log",
    "job.lost": "Status job training tidak bisa diikuti.",
    "anim.preprocess": "Memindai baris, membersihkan kolom dan mengenkode fitur",
    "anim.train": "Melatih {model} dan mencari threshold dengan F1 terbaik",
    "anim.trainOf": "Melatih {model} ({i} dari {n}) dan mencari threshold dengan F1 terbaik",
    "anim.done": "Selesai. Membuka hasil…",
    "anim.pause": "Jeda animasi",
    "anim.play": "Putar animasi",
    "time.s": "{s} dtk",
    "time.ms": "{m} mnt {s} dtk",

    "runs.title": "Riwayat run",
    "runs.loading": "Memuat riwayat…",
    "runs.loadError": "Log eksperimen tidak bisa dimuat.",
    "runs.emptyTitle": "Belum ada run",
    "runs.emptyBody": "Setiap run training tercatat di sini beserta skor, plot dan file modelnya.",
    "runs.emptyCta": "Unggah dataset",
    "runs.lede.one": "1 run tercatat di {code}. Skor diukur pada 20% data hold-out tiap dataset, jadi bandingkan hanya run dengan target yang sama.",
    "runs.lede.many": "{n} run tercatat di {code}. Skor diukur pada 20% data hold-out tiap dataset, jadi bandingkan hanya run dengan target yang sama.",
    "runs.download": "Unduh log (CSV)",
    "runs.caption": "Daftar run training, bisa diurutkan per kolom",
    "runs.positive": "positif: {v}",

    "m.run": "Run", "m.target": "Target", "m.f1": "F1", "m.f1score": "Skor F1", "m.precision": "Presisi",
    "m.recall": "Recall", "m.prauc": "PR-AUC", "m.accuracy": "Akurasi", "m.threshold": "Threshold",

    "run.loading": "Memuat run…",
    "run.loadError": "Run ini tidak bisa dimuat.",
    "run.notFoundTitle": "Run tidak ditemukan",
    "run.notFoundBody": "{id} tidak ada di log eksperimen. Mungkin sudah dihapus.",
    "run.back": "Kembali ke semua run",
    "run.predicting": "Memprediksi {target}",
    "run.positiveClass": "kelas positif {pos}",
    "run.vs": "vs {neg}",
    "run.evaluatedOn": "dievaluasi pada {n} baris hold-out",
    "run.trainAnother": "Latih model lain",
    "run.tabs": "Detail run",
    "tab.evaluation": "Evaluasi", "tab.explain": "Penjelasan", "tab.threshold": "Threshold", "tab.files": "File",
    "fig.missing": "Plot ini tidak dibuat untuk run ini.",
    "fig.cm": "<strong>Confusion matrix</strong> pada threshold terpilih {t}: prediksi benar ada di diagonal, kesalahan di luar diagonal.",
    "fig.roc": "<strong>Kurva ROC.</strong> True positive rate terhadap false positive rate di semua threshold. Titik merah adalah titik operasi yang dipilih.",
    "fig.pr": "<strong>Kurva precision-recall.</strong> Lebih informatif daripada ROC saat kelas positif jarang. Garis putus-putus adalah classifier acak.",
    "fig.shap": "<strong>Ringkasan SHAP</strong> untuk sampel baris hold-out.",
    "eval.heading": "Arti angka ini untuk {pos}",
    "eval.p1": "Pada threshold <strong>{t}</strong>, {p} baris yang dilabeli <strong>{pos}</strong> oleh model memang {pos} (presisi).",
    "eval.p2": "Dari semua baris yang memang {pos}, model menemukan {r} (recall).",
    "eval.p3": "PR-AUC sebesar <strong>{v}</strong> merangkum seluruh kurva, tidak bergantung pada threshold.",
    "shap.heading": "Cara membaca plot ini",
    "shap.p1": "Setiap baris adalah satu fitur, diurutkan dari pengaruh terbesar terhadap model. Setiap titik adalah satu baris data hold-out.",
    "shap.p2": "Titik di kanan mendorong prediksi ke arah <strong>{pos}</strong>; titik di kiri menjauhkannya. Warna menunjukkan nilai fitur pada baris itu: merah tinggi, biru rendah.",
    "shap.p3": "Jika satu fitur sangat dominan dan skornya terlihat terlalu bagus, cek apakah fitur itu membocorkan jawaban (misalnya kolom yang dicatat setelah hasilnya terjadi).",
    "th.heading": "Kenapa {t}, bukan 0.50",
    "th.p1": "Model menghasilkan probabilitas. Sebuah baris dilabeli <strong>{pos}</strong> jika probabilitasnya minimal sebesar threshold.",
    "th.p2": "Pipeline mencoba setiap threshold dari 0.10 sampai 0.90 dengan langkah 0.01 pada data hold-out, lalu memilih yang skor F1-nya tertinggi.",
    "th.aria": "Skor F1 per threshold keputusan. F1 tertinggi {f} pada threshold {t}.",
    "th.axis": "Threshold keputusan",
    "th.marker": "F1 {f} di {t}",
    "th.none": "Tidak ada data sweep threshold untuk run ini.",
    "files.heading": "File",
    "files.params": "Parameter",
    "files.model": "Model terlatih. Muat dengan xgboost.XGBClassifier().load_model().",
    "files.metrics": "Semua metrik, label kelas, dan seluruh sweep threshold untuk run ini.",
    "files.download": "Unduh",
    "files.na": "Tidak tersedia untuk run ini.",
    "files.trainRows": "Baris training",
    "files.features": "Fitur setelah encoding",
    "pos.fallback": "kelas positif",
    "pos.short": "positif",

    "ds.modelHeading": "Pilih model",
    "ds.modelLede": "Setiap model memakai split yang sama, penanganan ketidakseimbangan kelas yang sama, dan threshold F1 masing-masing, jadi skornya bisa dibandingkan.",
    "ds.trainOne": "Latih model {model}",
    "ds.trainAll": "Latih dan bandingkan {n} model",
    "ds.modelLabel": "Model",
    "ds.allModels": "semua {n} model",
    "ds.excludeHeading": "Kecualikan kolom (opsional)",
    "ds.excludeLede": "Keluarkan kolom yang tidak boleh dipelajari model, misalnya nilai yang baru diketahui setelah hasilnya terjadi, seperti lama telepon penawaran. Kolom ID dan kolom konstan sudah dibuang otomatis.",
    "ds.excludeToggle": "Pilih kolom yang dikecualikan",
    "ds.excludeCount.none": "belum ada yang dipilih",
    "ds.excludeCount": "{n} dipilih",
    "ds.excludedLabel": "Dikecualikan",
    "ds.excludeTarget": "target",
    "job.excluded": "dikecualikan: {cols}",
    "files.excluded": "Kolom yang dikecualikan",
    "files.none": "tidak ada",
    "model.compare": "Bandingkan semua model",
    "model.compare.desc": "Melatih {n} model satu per satu pada split yang sama, lalu menampilkannya berdampingan.",
    "model.xgboost.desc": "Pohon gradient boosting. Pilihan awal yang kuat untuk data tabel.",
    "model.random_forest.desc": "Banyak pohon keputusan independen yang dirata-rata. Stabil dan tidak mudah overfit.",
    "model.logistic_regression.desc": "Baseline linear. Menunjukkan apakah model yang rumit memang diperlukan.",
    "model.hist_gradient_boosting.desc": "Gradient boosting berbasis histogram dari scikit-learn. SHAP diestimasi dengan sampling.",
    "model.lightgbm.desc": "Gradient boosting leaf-wise. Cepat untuk dataset besar.",
    "model.catboost.desc": "Gradient boosting dengan ordered boosting. Sering bagus tanpa tuning.",
    "m.model": "Model",
    "job.viewFinished.one": "Buka run yang sudah selesai",
    "job.viewFinished.many": "Bandingkan {n} run yang sudah selesai",
    "runs.select": "Pilih",
    "runs.selectRun": "Pilih {id} untuk dibandingkan",
    "runs.compare": "Bandingkan yang dipilih ({n})",
    "runs.compareHint": "Centang dua run atau lebih untuk membandingkannya berdampingan.",
    "run.modelMeta": "model {model}",
    "cmp.title": "Perbandingan model",
    "cmp.loading": "Memuat perbandingan…",
    "cmp.heading": "Membandingkan {n} model untuk {target}",
    "cmp.headingMixed": "Membandingkan {n} run",
    "cmp.lede": "Diurutkan berdasarkan skor F1. Run yang dilatih dalam job yang sama memakai split 80/20 yang sama, jadi skornya bisa langsung dibandingkan.",
    "cmp.mixed": "Run ini memprediksi target yang berbeda, jadi skornya tidak bisa dibandingkan satu sama lain.",
    "cmp.missing.one": "1 run tidak bisa dimuat dan tidak ditampilkan.",
    "cmp.missing.many": "{n} run tidak bisa dimuat dan tidak ditampilkan.",
    "cmp.notEnough": "Pilih minimal dua run untuk dibandingkan.",
    "cmp.bestF1": "F1 terbaik",
    "cmp.bestPr": "PR-AUC terbaik",
    "cmp.chart": "Skor F1 per model",
    "cmp.caption": "Perbandingan model, diurutkan berdasarkan skor F1",
    "files.modelPrefix": "Model {model} terlatih.",
    "files.load.json": "Muat dengan xgboost.XGBClassifier().load_model().",
    "files.load.joblib": "Muat dengan joblib.load(). Format ini memakai pickle: hanya buka file yang Anda percaya.",
    "files.load.txt": "Muat dengan lightgbm.Booster(model_file=...).",
    "files.load.cbm": "Muat dengan catboost.CatBoostClassifier().load_model().",
    "files.classWeight": "class_weight",
    "files.balanced": "balanced",
    "shap.note.hist_gradient_boosting": "Untuk HistGradientBoosting, nilai SHAP diestimasi dengan sampling (maksimal 300 baris), jadi hasilnya perkiraan.",
    "shap.note.logistic_regression": "Untuk logistic regression, nilai SHAP dihitung pada fitur yang sudah distandarkan; warnanya tetap menunjukkan nilai asli.",
  },
};

function initialLang() {
  try {
    const saved = localStorage.getItem("lang");
    if (saved && I18N[saved]) return saved;
  } catch { /* storage blocked */ }
  return (navigator.language || "").toLowerCase().startsWith("id") ? "id" : "en";
}

let lang = initialLang();

/** Translate a key; {name} placeholders are replaced with already-escaped values. */
function t(key, vars = {}) {
  const template = (I18N[lang] && I18N[lang][key]) ?? I18N.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name) => (name in vars ? String(vars[name]) : `{${name}}`));
}

const locale = () => (lang === "id" ? "id-ID" : "en-GB");

/* ------------------------------------------------------------------ state + helpers */

const view = document.getElementById("view");
const main = document.getElementById("main");
const POLL_MS = 1000;
const ANIM_BARS = 28;

let pollTimer = null;
let firstRoute = true;
let animationPaused = false;
let modelsCache = null;    // [{key, label, default}] from /api/models
let selectedModel = null;  // a model key, or "compare" for all models
let excludedColumns = new Set();  // columns left out of training, per dataset
let excludedFor = null;           // which dataset the set above belongs to

async function loadModels() {
  if (!modelsCache) modelsCache = await api("/api/models");
  return modelsCache;
}

const modelLabel = (key) => ((modelsCache || []).find((m) => m.key === key) || {}).label || key;

/** Fetch JSON from the API; throws Error(message) with .status on failure. */
async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    const err = new Error(t("error.network"));
    err.status = 0;
    throw err;
  }
  const isJson = (res.headers.get("content-type") || "").includes("application/json");
  const body = isJson ? await res.json() : null;
  if (!res.ok) {
    const detail = body && body.detail;
    const message = typeof detail === "string" ? detail : `${res.status} ${res.statusText}`;
    const err = new Error(message.charAt(0).toUpperCase() + message.slice(1));
    err.status = res.status;
    throw err;
  }
  return body;
}

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const num = (value, digits = 3) =>
  value === null || value === undefined || Number.isNaN(Number(value)) ? "n/a" : Number(value).toFixed(digits);

const int = (value) => Number(value).toLocaleString("en-US");

const pct = (part, whole) => (whole ? `${((100 * part) / whole).toFixed(1)}%` : "0%");

const pctOf = (ratio) => (ratio === null || ratio === undefined ? "n/a" : `${(100 * ratio).toFixed(1)}%`);

function compact(value) {
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `${(value / 1e3).toFixed(1)}k`;
  if (abs >= 100 || Number.isInteger(value)) return String(Math.round(value));
  return value.toFixed(2);
}

function bytes(size) {
  if (size >= 1048576) return `${(size / 1048576).toFixed(1)} MB`;
  if (size >= 1024) return `${(size / 1024).toFixed(0)} KB`;
  return `${size} B`;
}

/** run_20261108_153022 -> "8 Nov 2026, 15:30" (time is the server's clock). */
function runDate(runId) {
  const m = /^run_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(runId || "");
  if (!m) return runId;
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
  return d.toLocaleString(locale(), { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

function elapsed(fromIso, toIso) {
  if (!fromIso) return "";
  const seconds = Math.max(0, Math.round(((toIso ? new Date(toIso) : new Date()) - new Date(fromIso)) / 1000));
  return seconds < 60 ? t("time.s", { s: seconds }) : t("time.ms", { m: Math.floor(seconds / 60), s: seconds % 60 });
}

function setTitle(text) {
  document.title = text ? `${text} · Mini-AutoML` : "Mini-AutoML";
}

function render(html) {
  view.innerHTML = html;
}

function clearJobView() {
  delete view.dataset.jobSignature;
}

function renderLoading(text) {
  render(`<p class="state" role="status">${esc(text)}</p>`);
}

function renderError(title, error, retry) {
  render(`
    <div class="alert" role="alert">
      <p><strong>${esc(title)}</strong></p>
      <p>${esc(error.message)}</p>
      <div class="alert-actions"><button class="btn btn-secondary" type="button" data-action="retry">${esc(t("retry"))}</button></div>
    </div>`);
  view.querySelector('[data-action="retry"]').addEventListener("click", retry);
}

function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}

async function refreshRunsCount() {
  try {
    const runs = await api("/api/runs");
    document.getElementById("runs-count").textContent = runs.length ? String(runs.length) : "";
  } catch {
    document.getElementById("runs-count").textContent = "";
  }
}

/* ------------------------------------------------------------------ router */

const routes = [
  [/^#\/compare\/(run_\d{8}_\d{6}(?:,run_\d{8}_\d{6})*)$/, (ids) => renderCompare(ids.split(","))],
  [/^#\/runs\/(run_\d{8}_\d{6})$/, (id) => renderRun(id)],
  [/^#\/runs\/?$/, () => renderRuns()],
  [/^#\/new\/?$/, () => renderNew()],
];

function router() {
  stopPolling();
  clearJobView();
  const hash = location.hash || "#/new";
  const section = hash.startsWith("#/runs") || hash.startsWith("#/compare") ? "runs" : hash.startsWith("#/new") ? "new" : "";
  document.querySelectorAll("[data-nav]").forEach((link) => {
    if (link.dataset.nav === section) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });

  const match = routes.find(([pattern]) => pattern.test(hash));
  if (match) match[1](...hash.match(match[0]).slice(1));
  else renderNotFound();

  // Move focus to the new content on navigation (not on first load), for keyboard users.
  if (!firstRoute) main.focus({ preventScroll: true });
  firstRoute = false;
  window.scrollTo(0, 0);
}

function renderNotFound() {
  setTitle(t("notFound.title"));
  render(`
    <div class="empty">
      <h1>${esc(t("notFound.title"))}</h1>
      <p>${esc(t("notFound.body"))}</p>
      <a class="btn btn-primary" href="#/new">${esc(t("notFound.cta"))}</a>
    </div>`);
}

/* ------------------------------------------------------------------ new run */

async function renderNew() {
  setTitle(t("nav.new"));
  renderLoading(t("new.loading"));
  let dataset = null;
  let current = null;
  try {
    [dataset, current] = await Promise.all([
      api("/api/dataset").catch((err) => (err.status === 404 ? null : Promise.reject(err))),
      api("/api/jobs/current"),
      loadModels(),
    ]);
  } catch (err) {
    renderError(t("new.loadError"), err, renderNew);
    return;
  }
  if (current.job) renderJob(current.job, dataset);
  else if (dataset) renderDataset(dataset);
  else renderUpload();
}

function renderUpload(previous = null) {
  render(`
    <div class="page-head">
      <div>
        <h1>${esc(t("upload.title"))}</h1>
        <p class="lede">${esc(t("upload.lede"))}</p>
      </div>
      ${previous ? `<button class="btn btn-secondary" type="button" data-action="cancel">${esc(t("upload.keep"))}</button>` : ""}
    </div>
    <form class="upload-form" novalidate>
      <label class="dropzone" for="file-input">
        <span class="dz-title">${esc(t("upload.drop"))}</span>
        <span class="dz-hint" id="file-hint">${esc(t("upload.hint"))}</span>
        <input id="file-input" name="file" type="file" accept=".csv,.tsv,.txt,text/csv" aria-describedby="file-hint">
      </label>
      <p class="upload-status" id="upload-status" role="status" aria-live="polite"></p>
    </form>`);

  const input = view.querySelector("#file-input");
  const zone = view.querySelector(".dropzone");
  input.addEventListener("change", () => input.files[0] && uploadFile(input.files[0]));
  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, () => zone.classList.remove("is-dragging")));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
  view.querySelector("form").addEventListener("submit", (e) => e.preventDefault());
  const cancel = view.querySelector('[data-action="cancel"]');
  if (cancel) cancel.addEventListener("click", () => renderDataset(previous));
}

async function uploadFile(file) {
  const status = view.querySelector("#upload-status");
  const input = view.querySelector("#file-input");
  status.classList.remove("is-error");
  if (!/\.(csv|tsv|txt)$/i.test(file.name)) {
    status.classList.add("is-error");
    status.textContent = t("upload.badType", { name: file.name });
    return;
  }
  input.disabled = true;
  status.textContent = t("upload.progress", { name: file.name, size: bytes(file.size) });
  const form = new FormData();
  form.append("file", file);
  try {
    const dataset = await api("/api/dataset", { method: "POST", body: form });
    renderDataset(dataset);
  } catch (err) {
    input.disabled = false;
    input.value = "";
    status.classList.add("is-error");
    status.textContent = err.message;
  }
}

function columnViz(col) {
  if (col.id_like) return "";  // a distribution of unique IDs says nothing; the note explains it
  if (col.histogram) {
    const h = col.histogram;
    const peak = Math.max(...h.counts, 1);
    const bars = h.counts.map((c) => `<span style="height:${Math.max(2, (100 * c) / peak)}%"></span>`).join("");
    return `
      <div class="hist" aria-hidden="true">${bars}</div>
      <div class="hist-range"><span>${compact(h.min)}</span><span>${esc(t("col.mean", { v: compact(h.mean) }))}</span><span>${compact(h.max)}</span></div>`;
  }
  const total = col.top_values.reduce((s, v) => s + v.count, 0) + col.other_count;
  const rows = col.top_values.map((v) => `
      <li><span class="label" title="${esc(v.value)}">${esc(v.value)}</span><span class="pct">${pct(v.count, total)}</span>
        <span class="bar" aria-hidden="true"><span style="width:${(100 * v.count) / (total || 1)}%"></span></span></li>`).join("");
  const others = col.n_unique - col.top_values.length;
  const other = col.other_count
    ? `<li><span class="label">${esc(t(others === 1 ? "col.other.one" : "col.other.many", { n: int(others) }))}</span><span class="pct">${pct(col.other_count, total)}</span></li>`
    : "";
  return `<ul class="tops">${rows}${other}</ul>`;
}

function columnCard(col, index, selected) {
  const disabled = !col.binary;
  const kind = col.kind === "numeric" ? t("col.numeric") : t("col.text");
  const notes = [];
  if (col.id_like) notes.push(t("col.idNote"));
  if (col.n_missing) notes.push(t("col.missing", { n: int(col.n_missing) }));
  return `
    <label class="col-card${disabled ? " is-disabled" : ""}${selected ? " is-selected" : ""}"
           data-binary="${col.binary}" for="col-${index}">
      <input type="radio" name="target" id="col-${index}" value="${esc(col.name)}"
             ${disabled ? "disabled" : ""} ${selected ? "checked" : ""}
             aria-describedby="col-${index}-stats">
      <span class="col-head">
        <span class="col-type" title="${esc(kind)}" aria-hidden="true">${col.kind === "numeric" ? "#" : "A"}</span>
        <span class="col-name">${esc(col.name)}</span>
      </span>
      ${columnViz(col)}
      <span class="col-stats" id="col-${index}-stats">${esc(kind)} · ${esc(t("col.distinct", { n: int(col.n_unique) }))} · ${esc(col.binary ? t("col.canBeTarget") : t("col.notBinary"))}</span>
      ${notes.length ? `<span class="col-note">${esc(notes.join(" · "))}</span>` : ""}
      ${selected ? `<span class="col-note selected-note">${esc(t("col.selected"))}</span>` : ""}
    </label>`;
}

function renderDataset(dataset, { target = null, notice = null } = {}) {
  const binaryCount = dataset.columns.filter((c) => c.binary).length;
  const uploaded = new Date(dataset.uploaded_at).toLocaleString(locale(), { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  const onlyBinary = view.dataset.onlyBinary === "true";
  const sepKey = `sep.${dataset.separator}`;
  const sepName = I18N.en[sepKey] ? t(sepKey) : t("sep.other");
  const models = modelsCache && modelsCache.length ? modelsCache : [{ key: "xgboost", label: "XGBoost", default: true }];
  if (!selectedModel || (selectedModel !== "compare" && !models.some((m) => m.key === selectedModel))) {
    selectedModel = (models.find((m) => m.default) || models[0]).key;
  }
  const datasetKey = `${dataset.filename}|${dataset.uploaded_at}`;
  if (excludedFor !== datasetKey) {  // a new upload starts with nothing excluded
    excludedColumns = new Set();
    excludedFor = datasetKey;
  }
  const excludeItems = dataset.columns.map((c) => `
    <label class="check exclude-item">
      <input type="checkbox" value="${esc(c.name)}" ${excludedColumns.has(c.name) ? "checked" : ""}>
      <span>${esc(c.name)}<span class="exclude-note"></span></span>
    </label>`).join("");
  const modelOption = (key, label, desc, cls = "") => `
    <label class="model-option ${cls}${key === selectedModel ? " is-selected" : ""}" for="model-${key}">
      <input type="radio" name="model" id="model-${key}" value="${key}" ${key === selectedModel ? "checked" : ""}>
      <span class="model-name">${esc(label)}</span>
      <span class="model-desc">${esc(desc)}</span>
    </label>`;

  render(`
    ${notice || ""}
    <div class="page-head">
      <div>
        <p class="breadcrumb">${esc(t("ds.crumb"))}</p>
        <h1>${esc(dataset.filename)}</h1>
        <p class="meta">
          <span><strong>${int(dataset.n_rows)}</strong> ${esc(t("ds.rows"))}</span>
          <span><strong>${int(dataset.n_cols)}</strong> ${esc(t("ds.columns"))}</span>
          <span>${esc(t("ds.separated", { sep: sepName, enc: dataset.encoding }))}</span>
          <span>${esc(t("ds.uploaded", { when: uploaded }))}</span>
        </p>
      </div>
      <button class="btn btn-secondary" type="button" data-action="replace">${esc(t("ds.replace"))}</button>
    </div>

    <section class="section" aria-labelledby="target-heading">
      <h2 id="target-heading">${esc(t("ds.heading"))}</h2>
      <p>${esc(binaryCount ? t("ds.binaryCount", { n: binaryCount, total: dataset.n_cols }) : t("ds.noBinary"))}</p>
      <div class="toolbar">
        <label class="check"><input type="checkbox" id="only-binary" ${onlyBinary ? "checked" : ""}> ${esc(t("ds.onlyBinary"))}</label>
      </div>
      <fieldset style="border:0;padding:0;margin:0">
        <legend class="visually-hidden">${esc(t("ds.legend"))}</legend>
        <div class="col-grid">
          ${dataset.columns.map((c, i) => columnCard(c, i, c.name === target)).join("")}
        </div>
      </fieldset>
    </section>

    <section class="section" aria-labelledby="exclude-heading">
      <h2 id="exclude-heading">${esc(t("ds.excludeHeading"))}</h2>
      <p>${esc(t("ds.excludeLede"))}</p>
      <details class="exclude-details" ${excludedColumns.size ? "open" : ""}>
        <summary>${esc(t("ds.excludeToggle"))} <span class="exclude-count" aria-live="polite"></span></summary>
        <fieldset class="exclude-grid">
          <legend class="visually-hidden">${esc(t("ds.excludeHeading"))}</legend>
          ${excludeItems}
        </fieldset>
      </details>
    </section>

    <section class="section" aria-labelledby="model-heading">
      <h2 id="model-heading">${esc(t("ds.modelHeading"))}</h2>
      <p>${esc(t("ds.modelLede"))}</p>
      <fieldset style="border:0;padding:0;margin:1rem 0 0">
        <legend class="visually-hidden">${esc(t("ds.modelHeading"))}</legend>
        <div class="model-grid">
          ${models.map((m) => modelOption(m.key, m.label, t(`model.${m.key}.desc`))).join("")}
          ${models.length > 1 ? modelOption("compare", t("model.compare"), t("model.compare.desc", { n: models.length }), "is-compare") : ""}
        </div>
      </fieldset>
    </section>

    <div class="actionbar">
      <p id="target-summary" aria-live="polite"></p>
      <button class="btn btn-primary" type="button" id="train-btn" disabled></button>
    </div>`);

  const grid = view.querySelector(".col-grid");
  const modelGrid = view.querySelector(".model-grid");
  const trainBtn = view.querySelector("#train-btn");
  const summary = view.querySelector("#target-summary");
  const chosenModels = () => (selectedModel === "compare" ? models.map((m) => m.key) : [selectedModel]);
  const excludeGrid = view.querySelector(".exclude-grid");
  const excludeCount = view.querySelector(".exclude-count");

  /** The target can never be excluded: disable its checkbox and drop it from the set. */
  function syncExclude(target) {
    excludeGrid.querySelectorAll("input").forEach((box) => {
      const isTarget = box.value === target;
      if (isTarget) { box.checked = false; excludedColumns.delete(box.value); }
      box.disabled = isTarget;
      box.closest(".exclude-item").classList.toggle("is-target", isTarget);
      box.closest(".exclude-item").querySelector(".exclude-note").textContent = isTarget ? ` (${t("ds.excludeTarget")})` : "";
    });
    excludeCount.textContent = excludedColumns.size
      ? `· ${t("ds.excludeCount", { n: excludedColumns.size })}`
      : `· ${t("ds.excludeCount.none")}`;
  }

  function applyFilter(checked) {
    view.dataset.onlyBinary = String(checked);
    grid.querySelectorAll(".col-card").forEach((card) => {
      card.hidden = checked && card.dataset.binary !== "true";
    });
  }

  function updateSelection() {
    const checked = grid.querySelector('input[name="target"]:checked');
    syncExclude(checked ? checked.value : null);
    grid.querySelectorAll(".col-card").forEach((card) => {
      const on = card.querySelector("input").checked;
      card.classList.toggle("is-selected", on);
      const note = card.querySelector(".selected-note");
      if (on && !note) card.insertAdjacentHTML("beforeend", `<span class="col-note selected-note">${esc(t("col.selected"))}</span>`);
      if (!on && note) note.remove();
    });
    modelGrid.querySelectorAll(".model-option").forEach((option) => {
      option.classList.toggle("is-selected", option.querySelector("input").checked);
    });
    const compare = selectedModel === "compare";
    trainBtn.textContent = compare
      ? t("ds.trainAll", { n: models.length })
      : t("ds.trainOne", { model: modelLabel(selectedModel) });
    trainBtn.disabled = !checked;
    if (!checked) {
      summary.textContent = t("ds.selectHint");
      return;
    }
    const col = dataset.columns.find((c) => c.name === checked.value);
    const total = col.top_values.reduce((s, v) => s + v.count, 0);
    const split = col.top_values.map((v) => `${esc(v.value)} ${pct(v.count, total)}`).join(", ");
    const modelText = compare ? t("ds.allModels", { n: models.length }) : modelLabel(selectedModel);
    const excludedText = excludedColumns.size
      ? ` · ${esc(t("ds.excludedLabel"))}: <strong>${esc([...excludedColumns].join(", "))}</strong>`
      : "";
    summary.innerHTML = `${esc(t("ds.target"))}: <strong>${esc(col.name)}</strong> (${split}) · ${esc(t("ds.modelLabel"))}: <strong>${esc(modelText)}</strong>${excludedText}`;
  }

  modelGrid.addEventListener("change", (e) => {
    selectedModel = e.target.value;
    updateSelection();
  });
  excludeGrid.addEventListener("change", (e) => {
    if (e.target.checked) excludedColumns.add(e.target.value);
    else excludedColumns.delete(e.target.value);
    updateSelection();
  });

  view.querySelector('[data-action="replace"]').addEventListener("click", () => renderUpload(dataset));
  view.querySelector("#only-binary").addEventListener("change", (e) => applyFilter(e.target.checked));
  grid.addEventListener("change", updateSelection);
  trainBtn.addEventListener("click", async () => {
    const checked = grid.querySelector('input[name="target"]:checked');
    if (!checked) return;
    trainBtn.disabled = true;
    summary.textContent = t("ds.starting");
    try {
      const job = await api("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: checked.value, models: chosenModels(), exclude: [...excludedColumns] }),
      });
      renderJob(job, dataset);
    } catch (err) {
      trainBtn.disabled = false;
      summary.innerHTML = `<span style="color:var(--danger);font-weight:600">${esc(err.message)}</span>`;
    }
  });

  applyFilter(onlyBinary);
  updateSelection();
}

/* ------------------------------------------------------------------ job progress + animation */

function stepLabel(step) {
  if (step.key === "preprocess") return t("step.preprocess");
  if (step.model) return t("step.train", { model: modelLabel(step.model) });
  return step.key === "train" ? t("step.train", { model: "XGBoost" }) : step.label;  // older single-model jobs
}

/** Link to whatever finished before a job stopped (one run page, or a comparison). */
function finishedRunsLink(job) {
  const ids = job.run_ids || [];
  if (!ids.length) return "";
  const href = ids.length > 1 ? `#/compare/${ids.join(",")}` : `#/runs/${ids[0]}`;
  const text = ids.length > 1 ? t("job.viewFinished.many", { n: ids.length }) : t("job.viewFinished.one");
  return `<a class="btn btn-secondary" href="${href}">${esc(text)}</a>`;
}

function stepState(step, job) {
  if (step.status === "running") return t("job.running", { t: elapsed(step.started_at) });
  if (step.status === "done") return t("job.done", { t: elapsed(step.started_at, step.finished_at) });
  if (step.status === "failed") return t("job.failed");
  return job.status === "failed" ? t("job.skipped") : t("job.waiting");
}

/**
 * Training animation. Purpose: feedback during a 10 to 60 second wait that also shows what
 * the current stage does. Built from the UI's identity motif (small blue data bars):
 *   preprocess -> a highlight sweeps across uneven bars (rows being scanned and cleaned)
 *   train      -> bars grow left to right into a rising curve (boosted trees improving the fit)
 *   done       -> bars settle at full height
 * Pausable (it runs longer than 5 s) and static under prefers-reduced-motion.
 */
function trainingAnimation(phase, caption) {
  const bars = Array.from({ length: ANIM_BARS }, (_, i) => {
    const uneven = 30 + ((i * 37) % 61);                          // deterministic "raw data" heights
    const rising = Math.round(18 + 78 * (1 - Math.exp(-(i + 1) / 7)));  // learning-curve shape
    return `<span style="--i:${i};--h-raw:${uneven}%;--h-fit:${rising}%"></span>`;
  }).join("");
  return `
    <div class="train-anim${animationPaused ? " is-paused" : ""}" data-phase="${phase}">
      <div class="anim-bars" aria-hidden="true">${bars}</div>
      <div class="anim-foot">
        <p class="anim-caption">${esc(caption)}</p>
        ${phase === "done" ? "" : `<button class="btn btn-secondary anim-toggle" type="button">${esc(t(animationPaused ? "anim.play" : "anim.pause"))}</button>`}
      </div>
    </div>`;
}

function renderJob(job, dataset) {
  setTitle(t("job.titleRunning", { target: job.target }));
  const failed = job.status === "failed";
  const succeeded = job.status === "succeeded";
  const signature = `${lang}:${job.id}:${job.status}:${job.steps.map((s) => s.status).join(",")}`;
  if (view.dataset.jobSignature === signature && job.status === "running") {
    // Nothing changed state: only refresh the timers, so focus, open logs and the animation stay put.
    view.querySelectorAll(".step-state").forEach((el, i) => { el.textContent = stepState(job.steps[i], job); });
    const timer = view.querySelector("[data-elapsed]");
    if (timer) timer.textContent = t("job.elapsed", { t: elapsed(job.started_at, job.finished_at) });
    schedulePoll(job, dataset);
    return;
  }
  const active = document.activeElement;
  const focusedStep = active && active.closest("details.log") ? active.closest("details.log").dataset.step : null;
  const focusedToggle = active && active.classList.contains("anim-toggle");
  const openLogs = new Set([...view.querySelectorAll("details.log[open]")].map((d) => d.dataset.step));
  const running = job.steps.find((s) => s.status === "running");
  const phase = succeeded ? "done" : running && running.key !== "preprocess" ? "train" : "preprocess";
  const trainSteps = job.steps.filter((s) => s.key !== "preprocess");
  let caption = t(`anim.${phase}`);
  if (phase === "train") {
    const model = modelLabel(running.model || "xgboost");
    caption = trainSteps.length > 1
      ? t("anim.trainOf", { model, i: trainSteps.indexOf(running) + 1, n: trainSteps.length })
      : t("anim.train", { model });
  }
  const title = failed ? "job.titleStopped" : succeeded ? "job.titleDone" : "job.titleRunning";
  const timeKey = failed ? "job.stoppedAfter" : "job.elapsed";

  render(`
    ${failed ? `
      <div class="alert" role="alert">
        <p><strong>${esc(t("job.failedTitle"))}</strong> ${esc(job.error || "")}</p>
        <div class="alert-actions">
          ${dataset ? `<button class="btn btn-primary" type="button" data-action="back">${esc(t("job.chooseAnother"))}</button>` : ""}
          <a class="btn btn-secondary" href="#/runs">${esc(t("job.seeRuns"))}</a>
          ${finishedRunsLink(job)}
        </div>
      </div>` : ""}
    <div class="page-head">
      <div>
        <p class="breadcrumb">${esc(job.dataset)}</p>
        <h1>${esc(t(title, { target: job.target }))}</h1>
        <p class="meta"><span>${esc(t("job.started", { time: new Date(job.started_at).toLocaleTimeString(locale()) }))}</span>
          <span data-elapsed>${esc(t(timeKey, { t: elapsed(job.started_at, job.finished_at) }))}</span>
          ${job.exclude && job.exclude.length ? `<span>${esc(t("job.excluded", { cols: job.exclude.join(", ") }))}</span>` : ""}</p>
      </div>
    </div>
    ${failed ? "" : trainingAnimation(phase, caption)}
    <ol class="steps">
      ${job.steps.map((step, i) => `
        <li class="is-${esc(step.status)}">
          <div class="step-row">
            <span class="step-num" aria-hidden="true">${i + 1}</span>
            <span class="step-label">${esc(stepLabel(step))}</span>
            <span class="step-state">${esc(stepState(step, job))}</span>
          </div>
          ${step.log ? `
            <details class="log" data-step="${esc(step.key)}" ${step.status === "failed" || openLogs.has(step.key) ? "open" : ""}>
              <summary>${esc(t("job.showLog"))}</summary>
              <pre class="log-text">${esc(step.log)}</pre>
            </details>` : ""}
        </li>`).join("")}
    </ol>
    <p class="visually-hidden" role="status">${esc(job.steps.map((s, i) => `${t("step.n", { n: i + 1 })}: ${stepState(s, job)}`).join(". "))}</p>`);

  view.dataset.jobSignature = signature;
  const back = view.querySelector('[data-action="back"]');
  if (back) back.addEventListener("click", () => { clearJobView(); renderDataset(dataset, { target: null }); });
  const toggle = view.querySelector(".anim-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      animationPaused = !animationPaused;
      view.querySelector(".train-anim").classList.toggle("is-paused", animationPaused);
      toggle.textContent = t(animationPaused ? "anim.play" : "anim.pause");
    });
  }
  if (focusedStep) {
    const summary = view.querySelector(`details.log[data-step="${focusedStep}"] summary`);
    if (summary) summary.focus();
  } else if (focusedToggle && toggle) {
    toggle.focus();
  } else if (failed && back) {
    back.focus();
  }

  if (job.status === "running") {
    schedulePoll(job, dataset);
  } else if (succeeded) {
    clearJobView();
    refreshRunsCount();
    // Hold the finished state briefly so the completion is visible, then open the run.
    const ids = job.run_ids && job.run_ids.length ? job.run_ids : job.run_id ? [job.run_id] : [];
    pollTimer = setTimeout(() => {
      location.hash = ids.length > 1 ? `#/compare/${ids.join(",")}` : ids.length ? `#/runs/${ids[0]}` : "#/runs";
    }, 1200);
  }
}

function schedulePoll(job, dataset) {
  pollTimer = setTimeout(async () => {
    try {
      renderJob(await api(`/api/jobs/${job.id}`), dataset);
    } catch (err) {
      clearJobView();
      renderError(t("job.lost"), err, renderNew);
    }
  }, POLL_MS);
}

/* ------------------------------------------------------------------ runs list */

const RUN_COLUMNS = [
  { key: "run_id", label: "m.run", numeric: false },
  { key: "target", label: "m.target", numeric: false },
  { key: "model_label", label: "m.model", numeric: false },
  { key: "f1_score", label: "m.f1", numeric: true },
  { key: "precision", label: "m.precision", numeric: true },
  { key: "recall", label: "m.recall", numeric: true },
  { key: "pr_auc", label: "m.prauc", numeric: true },
  { key: "accuracy", label: "m.accuracy", numeric: true },
  { key: "best_threshold", label: "m.threshold", numeric: true },
];

async function renderRuns() {
  setTitle(t("runs.title"));
  renderLoading(t("runs.loading"));
  let runs;
  try {
    runs = await api("/api/runs");
  } catch (err) {
    renderError(t("runs.loadError"), err, renderRuns);
    return;
  }
  document.getElementById("runs-count").textContent = runs.length ? String(runs.length) : "";

  if (!runs.length) {
    render(`
      <div class="page-head"><div><h1>${esc(t("runs.title"))}</h1></div></div>
      <div class="empty">
        <h2>${esc(t("runs.emptyTitle"))}</h2>
        <p>${esc(t("runs.emptyBody"))}</p>
        <a class="btn btn-primary" href="#/new">${esc(t("runs.emptyCta"))}</a>
      </div>`);
    return;
  }

  let sort = { key: "run_id", dir: "desc" };
  const selected = new Set();  // run ids ticked for comparison; survives re-sorting

  function rowsHtml() {
    const sorted = [...runs].sort((a, b) => {
      const x = a[sort.key], y = b[sort.key];
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      const cmp = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
      return sort.dir === "asc" ? cmp : -cmp;
    });
    return sorted.map((r) => `
      <tr>
        <td class="select"><input type="checkbox" class="run-check" value="${esc(r.run_id)}"
              aria-label="${esc(t("runs.selectRun", { id: r.run_id }))}" ${selected.has(r.run_id) ? "checked" : ""}></td>
        <td><a class="run-link" href="#/runs/${esc(r.run_id)}">${esc(r.run_id)}</a><span class="sub">${esc(runDate(r.run_id))}</span></td>
        <td>${esc(r.target)}${r.positive_class ? `<span class="sub">${esc(t("runs.positive", { v: r.positive_class }))}</span>` : ""}</td>
        <td>${esc(r.model_label)}</td>
        ${RUN_COLUMNS.slice(3).map((c) => `<td class="num">${num(r[c.key], c.key === "best_threshold" ? 2 : 3)}</td>`).join("")}
      </tr>`).join("");
  }

  function headHtml() {
    const selectHead = `<th scope="col" class="select"><span class="visually-hidden">${esc(t("runs.select"))}</span></th>`;
    return selectHead + RUN_COLUMNS.map((c) => {
      const active = sort.key === c.key;
      const ariaSort = active ? (sort.dir === "asc" ? "ascending" : "descending") : "none";
      const arrow = active ? (sort.dir === "asc" ? " ▲" : " ▼") : "";
      return `<th scope="col" class="${c.numeric ? "num" : ""}" aria-sort="${ariaSort}">
        <button class="sort-btn" type="button" data-key="${c.key}" data-active="${active}">${esc(t(c.label))}<span aria-hidden="true">${arrow}</span></button></th>`;
    }).join("");
  }

  const code = "<code>outputs/experiments_log.csv</code>";
  render(`
    <div class="page-head">
      <div>
        <h1>${esc(t("runs.title"))}</h1>
        <p class="lede">${t(runs.length === 1 ? "runs.lede.one" : "runs.lede.many", { n: runs.length, code })}</p>
      </div>
      <div class="head-actions">
        <a class="btn btn-secondary" href="/api/experiments_log.csv" download>${esc(t("runs.download"))}</a>
        <button class="btn btn-primary" type="button" id="compare-btn" disabled aria-describedby="compare-hint"></button>
      </div>
    </div>
    <p class="compare-hint" id="compare-hint">${esc(t("runs.compareHint"))}</p>
    <div class="table-wrap">
      <table>
        <caption class="visually-hidden">${esc(t("runs.caption"))}</caption>
        <thead><tr>${headHtml()}</tr></thead>
        <tbody>${rowsHtml()}</tbody>
      </table>
    </div>`);

  view.querySelector("thead").addEventListener("click", (e) => {
    const btn = e.target.closest(".sort-btn");
    if (!btn) return;
    const key = btn.dataset.key;
    const numeric = RUN_COLUMNS.find((c) => c.key === key).numeric;
    sort = sort.key === key ? { key, dir: sort.dir === "asc" ? "desc" : "asc" } : { key, dir: numeric || key === "run_id" ? "desc" : "asc" };
    view.querySelector("thead tr").innerHTML = headHtml();
    view.querySelector("tbody").innerHTML = rowsHtml();
    view.querySelector(`.sort-btn[data-key="${key}"]`).focus();
  });

  const compareBtn = view.querySelector("#compare-btn");
  const syncCompare = () => {
    compareBtn.textContent = t("runs.compare", { n: selected.size });
    compareBtn.disabled = selected.size < 2;
  };
  view.querySelector("tbody").addEventListener("change", (e) => {
    if (!e.target.classList.contains("run-check")) return;
    if (e.target.checked) selected.add(e.target.value);
    else selected.delete(e.target.value);
    syncCompare();
  });
  compareBtn.addEventListener("click", () => {
    // Keep the table order (newest first) so the comparison URL is stable.
    const ids = runs.map((r) => r.run_id).filter((id) => selected.has(id));
    if (ids.length >= 2) location.hash = `#/compare/${ids.join(",")}`;
  });
  syncCompare();
}

/* ------------------------------------------------------------------ comparison */

async function renderCompare(ids) {
  setTitle(t("cmp.title"));
  renderLoading(t("cmp.loading"));
  const unique = [...new Set(ids)];
  let results;
  try {
    results = await Promise.all(unique.map((id) =>
      api(`/api/runs/${id}`).catch((err) => (err.status === 404 ? null : Promise.reject(err)))));
  } catch (err) {
    renderError(t("cmp.title"), err, () => renderCompare(ids));
    return;
  }
  const runs = results.filter(Boolean);
  const missing = unique.length - runs.length;
  const crumb = `<p class="breadcrumb"><a href="#/runs">${esc(t("runs.title"))}</a> / <span>${esc(t("cmp.title"))}</span></p>`;

  if (runs.length < 2) {
    render(`${crumb}
      <div class="empty"><h1>${esc(t("cmp.title"))}</h1>
        <p>${esc(t("cmp.notEnough"))}</p>
        <a class="btn btn-primary" href="#/runs">${esc(t("run.back"))}</a></div>`);
    return;
  }

  runs.sort((a, b) => (b.f1_score ?? -1) - (a.f1_score ?? -1));
  const mixed = new Set(runs.map((r) => r.target)).size > 1;
  const bestF1 = runs[0].run_id;
  const bestPr = runs.reduce((a, b) => ((b.pr_auc ?? -1) > (a.pr_auc ?? -1) ? b : a)).run_id;
  const heading = mixed
    ? t("cmp.headingMixed", { n: runs.length })
    : t("cmp.heading", { n: runs.length, target: runs[0].target });

  const notes = [];
  if (mixed) notes.push(t("cmp.mixed"));
  if (missing) notes.push(t(missing === 1 ? "cmp.missing.one" : "cmp.missing.many", { n: missing }));

  const tag = (text) => `<span class="best-tag">${esc(text)}</span>`;
  const rows = runs.map((r) => `
    <tr${r.run_id === bestF1 ? ' class="is-best"' : ""}>
      <th scope="row">
        <a href="#/runs/${esc(r.run_id)}">${esc(r.model_label)}</a>
        ${r.run_id === bestF1 ? tag(t("cmp.bestF1")) : ""}${r.run_id === bestPr ? tag(t("cmp.bestPr")) : ""}
        <span class="sub">${esc(r.run_id)}${mixed ? ` · ${esc(r.target)}` : ""}</span>
      </th>
      <td class="num">${num(r.f1_score)}</td>
      <td class="num">${num(r.pr_auc)}</td>
      <td class="num">${num(r.precision)}</td>
      <td class="num">${num(r.recall)}</td>
      <td class="num">${num(r.accuracy)}</td>
      <td class="num">${num(r.best_threshold, 2)}</td>
    </tr>`).join("");

  const bars = runs.map((r) => `
    <li>
      <span class="cmp-name">${esc(r.model_label)}</span>
      <span class="cmp-track" aria-hidden="true"><span style="width:${Math.max(0, Math.min(1, r.f1_score || 0)) * 100}%"></span></span>
      <span class="cmp-val">${num(r.f1_score)}</span>
    </li>`).join("");

  render(`${crumb}
    <div class="page-head">
      <div>
        <h1>${esc(heading)}</h1>
        <p class="lede">${esc(t("cmp.lede"))}</p>
      </div>
    </div>
    ${notes.map((n) => `<p class="note">${esc(n)}</p>`).join("")}
    <section class="section" aria-labelledby="cmp-chart-heading">
      <h2 id="cmp-chart-heading" class="h3">${esc(t("cmp.chart"))}</h2>
      <ul class="cmp-bars">${bars}</ul>
    </section>
    <div class="table-wrap section">
      <table>
        <caption class="visually-hidden">${esc(t("cmp.caption"))}</caption>
        <thead><tr>
          <th scope="col">${esc(t("m.model"))}</th>
          <th scope="col" class="num">${esc(t("m.f1"))}</th>
          <th scope="col" class="num">${esc(t("m.prauc"))}</th>
          <th scope="col" class="num">${esc(t("m.precision"))}</th>
          <th scope="col" class="num">${esc(t("m.recall"))}</th>
          <th scope="col" class="num">${esc(t("m.accuracy"))}</th>
          <th scope="col" class="num">${esc(t("m.threshold"))}</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`);
}

/* ------------------------------------------------------------------ run detail */

function figure(run, name, caption, cls = "") {
  const file = run.files[name];
  if (!file) {
    return `<figure class="${cls}"><div class="figure-missing">${esc(t("fig.missing"))}</div>
      <figcaption>${caption}</figcaption></figure>`;
  }
  const dims = file.width ? `width="${file.width}" height="${file.height}"` : "";
  return `<figure class="${cls}"><div class="frame">
      <img src="/api/runs/${esc(run.run_id)}/files/${name}" ${dims} alt="${esc(caption.replace(/<[^>]+>/g, ""))}" loading="lazy" decoding="async">
    </div><figcaption>${caption}</figcaption></figure>`;
}

function thresholdChart(run) {
  const points = run.threshold_curve || [];
  if (points.length < 2) return `<div class="figure-missing">${esc(t("th.none"))}</div>`;
  const W = 640, H = 300, L = 44, R = 16, T = 16, B = 36;
  const xs = points.map((p) => p.threshold);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const x = (v) => L + ((v - xMin) / (xMax - xMin)) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.threshold).toFixed(1)},${y(p.f1).toFixed(1)}`).join("");
  const best = points.reduce((a, b) => (b.f1 > a.f1 ? b : a));
  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTicks = [0.1, 0.3, 0.5, 0.7, 0.9].filter((v) => v >= xMin - 1e-9 && v <= xMax + 1e-9);
  const vars = { f: num(best.f1), t: num(best.threshold, 2) };
  return `
    <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(t("th.aria", vars))}">
      <g class="grid">${yTicks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/>`).join("")}</g>
      <g class="axis">
        ${yTicks.map((v) => `<text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`).join("")}
        ${xTicks.map((v) => `<text x="${x(v)}" y="${H - 14}" text-anchor="middle">${v.toFixed(1)}</text>`).join("")}
        <text x="${(L + W - R) / 2}" y="${H - 1}" text-anchor="middle">${esc(t("th.axis"))}</text>
      </g>
      <path class="curve" d="${path}"/>
      <line class="marker-line" x1="${x(best.threshold)}" x2="${x(best.threshold)}" y1="${T}" y2="${H - B}"/>
      <circle class="marker-dot" cx="${x(best.threshold)}" cy="${y(best.f1)}" r="5"/>
      <text class="marker-text" x="${x(best.threshold) + (best.threshold > 0.7 ? -8 : 8)}" y="${y(best.f1) - 10}"
            text-anchor="${best.threshold > 0.7 ? "end" : "start"}">${esc(t("th.marker", vars))}</text>
    </svg>`;
}

const RUN_TABS = ["evaluation", "explain", "threshold", "files"];

function tabPanel(id, run) {
  const pos = esc(run.positive_class || t("pos.fallback"));
  const posShort = esc(run.positive_class || t("pos.short"));
  if (id === "evaluation") {
    return `<div class="figures">
      ${figure(run, "confusion_matrix.png", t("fig.cm", { t: num(run.best_threshold, 2) }))}
      ${figure(run, "roc_curve.png", t("fig.roc"))}
      ${figure(run, "pr_curve.png", t("fig.pr"))}
      <div class="prose">
        <h2 class="h3">${t("eval.heading", { pos })}</h2>
        <p>${t("eval.p1", { t: num(run.best_threshold, 2), p: pctOf(run.precision), pos: posShort })}</p>
        <p>${t("eval.p2", { pos: posShort, r: pctOf(run.recall) })}</p>
        <p>${t("eval.p3", { v: num(run.pr_auc) })}</p>
      </div>
    </div>`;
  }
  if (id === "explain") {
    return `<div class="split">
      ${figure(run, "shap_summary.png", t("fig.shap"))}
      <div class="prose">
        <h2 class="h3">${esc(t("shap.heading"))}</h2>
        <p>${esc(t("shap.p1"))}</p>
        <p>${t("shap.p2", { pos })}</p>
        <p>${esc(t("shap.p3"))}</p>
        ${I18N.en[`shap.note.${run.model}`] ? `<p class="note">${esc(t(`shap.note.${run.model}`))}</p>` : ""}
      </div>
    </div>`;
  }
  if (id === "threshold") {
    return `<div class="split">
      <figure>${thresholdChart(run)}</figure>
      <div class="prose">
        <h2 class="h3">${esc(t("th.heading", { t: num(run.best_threshold, 2) }))}</h2>
        <p>${t("th.p1", { pos: posShort })}</p>
        <p>${esc(t("th.p2"))}</p>
      </div>
    </div>`;
  }
  const modelFile = run.model_file || "xgboost_model.json";
  const loadKey = `files.load.${modelFile.split(".").pop()}`;
  const files = [
    [modelFile, `${t("files.modelPrefix", { model: run.model_label || "XGBoost" })} ${I18N.en[loadKey] ? t(loadKey) : ""}`],
    ["run_metrics.json", t("files.metrics")],
  ];
  // Runs made before model choice have no "hyperparameters" in run_metrics.json: use the log columns.
  const hp = run.hyperparameters;
  const params = hp
    ? Object.entries(hp).map(([k, v]) => [k, k === "scale_pos_weight" ? num(v) : v])
    : [["max_depth", run.max_depth], ["learning_rate", run.learning_rate],
       ["n_estimators", run.n_estimators], ["scale_pos_weight", num(run.scale_pos_weight)]];
  if (hp && !("scale_pos_weight" in hp)) params.push([t("files.classWeight"), t("files.balanced")]);
  params.push([t("files.excluded"), (run.excluded_columns || []).join(", ") || t("files.none")]);
  return `<div class="split">
    <div>
      <h2 class="h3" style="margin-bottom:0.75rem">${esc(t("files.heading"))}</h2>
      <ul class="files">
        ${files.map(([name, desc]) => run.files[name]
          ? `<li><span class="fname">${name}<span class="fdesc">${esc(desc)} ${bytes(run.files[name].size)}.</span></span>
               <a class="btn btn-secondary" href="/api/runs/${esc(run.run_id)}/files/${name}?download=true" download>${esc(t("files.download"))}</a></li>`
          : `<li><span class="fname">${name}<span class="fdesc">${esc(t("files.na"))}</span></span></li>`).join("")}
      </ul>
    </div>
    <div>
      <h2 class="h3" style="margin-bottom:0.75rem">${esc(t("files.params"))}</h2>
      <dl class="kv">
        ${params.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v ?? "n/a")}</dd>`).join("")}
        <dt>${esc(t("files.trainRows"))}</dt><dd>${run.n_train ? int(run.n_train) : "n/a"}</dd>
        <dt>${esc(t("files.features"))}</dt><dd>${run.n_features ? int(run.n_features) : "n/a"}</dd>
      </dl>
    </div>
  </div>`;
}

async function renderRun(runId) {
  setTitle(runId);
  renderLoading(t("run.loading"));
  let run;
  try {
    run = await api(`/api/runs/${runId}`);
  } catch (err) {
    if (err.status === 404) {
      render(`
        <p class="breadcrumb"><a href="#/runs">${esc(t("runs.title"))}</a></p>
        <div class="empty"><h1>${esc(t("run.notFoundTitle"))}</h1>
          <p>${t("run.notFoundBody", { id: `<code>${esc(runId)}</code>` })}</p>
          <a class="btn btn-primary" href="#/runs">${esc(t("run.back"))}</a></div>`);
    } else {
      renderError(t("run.loadError"), err, () => renderRun(runId));
    }
    return;
  }
  setTitle(`${run.target} · ${runId}`);

  const scores = [
    ["m.precision", num(run.precision)],
    ["m.recall", num(run.recall)],
    ["m.prauc", num(run.pr_auc)],
    ["m.accuracy", num(run.accuracy)],
    ["m.threshold", num(run.best_threshold, 2)],
  ];
  const f1 = run.f1_score;
  const positive = run.positive_class
    ? `<span>${t("run.positiveClass", { pos: `<strong>${esc(run.positive_class)}</strong>` })}${run.negative_class ? ` ${esc(t("run.vs", { neg: run.negative_class }))}` : ""}</span>`
    : "";

  render(`
    <p class="breadcrumb"><a href="#/runs">${esc(t("runs.title"))}</a> / <span>${esc(runId)}</span></p>
    <div class="page-head">
      <div>
        <h1>${esc(t("run.predicting", { target: run.target }))}</h1>
        <p class="meta">
          <span>${t("run.modelMeta", { model: `<strong>${esc(run.model_label || "XGBoost")}</strong>` })}</span>
          <span>${esc(runDate(runId))}</span>
          ${positive}
          ${run.n_test ? `<span>${t("run.evaluatedOn", { n: `<strong>${int(run.n_test)}</strong>` })}</span>` : ""}
        </p>
      </div>
      <a class="btn btn-secondary" href="#/new">${esc(t("run.trainAnother"))}</a>
    </div>

    <dl class="scores">
      <div class="score focal"><dt>${esc(t("m.f1score"))}</dt><dd>${num(f1)}
        <span class="score-bar" aria-hidden="true"><span style="width:${Math.max(0, Math.min(1, f1 || 0)) * 100}%"></span></span></dd></div>
      ${scores.map(([label, value]) => `<div class="score"><dt>${esc(t(label))}</dt><dd>${value}</dd></div>`).join("")}
    </dl>

    <div class="tabs" role="tablist" aria-label="${esc(t("run.tabs"))}">
      ${RUN_TABS.map((id, i) => `<button class="tab" type="button" role="tab" id="tab-${id}" aria-controls="panel-${id}"
          aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}">${esc(t(`tab.${id}`))}</button>`).join("")}
    </div>
    ${RUN_TABS.map((id, i) => `<section class="tabpanel" role="tabpanel" id="panel-${id}" aria-labelledby="tab-${id}" tabindex="0" ${i === 0 ? "" : "hidden"}>
      ${tabPanel(id, run)}</section>`).join("")}`);

  const tabs = [...view.querySelectorAll('[role="tab"]')];
  function select(tab) {
    tabs.forEach((tb) => {
      const on = tb === tab;
      tb.setAttribute("aria-selected", String(on));
      tb.tabIndex = on ? 0 : -1;
      view.querySelector(`#${tb.getAttribute("aria-controls")}`).hidden = !on;
    });
    tab.focus();
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (e) => {
      const moves = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 };
      if (!(e.key in moves)) return;
      e.preventDefault();
      select(tabs[(moves[e.key] + tabs.length) % tabs.length]);
    });
  });
}

/* ------------------------------------------------------------------ shell: theme, language, boot */

function applyStaticText() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  document.getElementById("footer-text").innerHTML = t("footer.text", { code: "<code>outputs/experiments_log.csv</code>" });
  document.querySelectorAll("[data-lang]").forEach((btn) => btn.setAttribute("aria-pressed", String(btn.dataset.lang === lang)));
  syncThemeButton();
}

function syncThemeButton() {
  const button = document.getElementById("theme-toggle");
  const dark = document.documentElement.dataset.theme === "dark";
  button.setAttribute("aria-pressed", String(dark));
  button.title = t(dark ? "theme.toLight" : "theme.toDark");
}

function initTheme() {
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* storage blocked: session only */ }
    syncThemeButton();
  });
}

function initLanguage() {
  document.querySelectorAll("[data-lang]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.lang === lang) return;
      lang = btn.dataset.lang;
      try { localStorage.setItem("lang", lang); } catch { /* storage blocked: session only */ }
      applyStaticText();
      // Re-render the current page in the new language, keeping keyboard focus on the switch.
      firstRoute = true;
      router();
      btn.focus();
    });
  });
}

initTheme();
initLanguage();
applyStaticText();
refreshRunsCount();
window.addEventListener("hashchange", router);
router();
