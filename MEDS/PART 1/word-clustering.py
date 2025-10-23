import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords

# --------------------
# 0️⃣ NLTK stopwords
# --------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --------------------
# 1️⃣ Load questions
# --------------------
DATA_FILE = r"C:\Users\titouan\OneDrive\Bureau\M2-MIND\MEDS\PART 1\WikiTableQuestions-1.0.2-compact\WikiTableQuestions\data\training.tsv"
df_questions = pd.read_csv(DATA_FILE, sep='\t')
questions = df_questions['utterance'].dropna().tolist()
print(f"Loaded {len(questions)} questions")

# --------------------
# 2️⃣ Load tables and linearize
# --------------------
CSV_DIR = r"C:\Users\titouan\OneDrive\Bureau\M2-MIND\MEDS\PART 1\WikiTableQuestions-1.0.2-compact\WikiTableQuestions\csv"

tables = []
table_names = []

for subfolder in os.listdir(CSV_DIR):
    subpath = os.path.join(CSV_DIR, subfolder)
    if not os.path.isdir(subpath):
        continue

    for fname in os.listdir(subpath):
        if not fname.endswith(".tsv") and not fname.endswith(".csv"):
            continue
        fpath = os.path.join(subpath, fname)
        try:
            t = pd.read_csv(fpath, sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
            # Linearize: join headers + rows into a single string
            table_text = " | ".join(t.columns) + " || " + " || ".join([" | ".join(map(str,row)) for row in t.values])
            tables.append(table_text)
            table_names.append(fpath)
        except Exception as e:
            print(f"⚠️ Error reading {fpath}: {e}")

print(f"Loaded {len(tables)} tables")

# --------------------
# 3️⃣ Compute embeddings
# --------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding questions...")
question_embeddings = model.encode(questions, show_progress_bar=True)

print("Embedding tables...")
table_embeddings = model.encode(tables, show_progress_bar=True)

# --------------------
# 4️⃣ Clustering
# --------------------
n_clusters = 10
print(f"Clustering into {n_clusters} clusters...")

q_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
q_labels = q_kmeans.fit_predict(question_embeddings)

t_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
t_labels = t_kmeans.fit_predict(table_embeddings)

# --------------------
# 5️⃣ Helper: get top words per cluster
# --------------------
def get_top_words(items, labels, top_n=3):
    cluster_top_words = []
    for cluster_id in range(len(set(labels))):
        cluster_items = [items[i] for i in range(len(items)) if labels[i] == cluster_id]
        words = []
        for item in cluster_items:
            tokens = [w.lower() for w in item.split() if w.isalpha() and w.lower() not in stop_words]
            words.extend(tokens)
        most_common = [w for w, _ in Counter(words).most_common(top_n)]
        cluster_top_words.append(most_common)
    return cluster_top_words

# --------------------
# 6️⃣ Bar chart for cluster counts
# --------------------
def plot_cluster_bar(labels, title, save_path):
    counts = Counter(labels)
    plt.figure(figsize=(8,5))
    plt.bar([f"Cluster {i}" for i in counts.keys()], counts.values(), color='skyblue', edgecolor='black')
    plt.title(title)
    plt.ylabel("Number of items")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# --------------------
# 7️⃣ Pie chart for clusters with top words
# --------------------
def plot_cluster_pie(labels, cluster_top_words, title, save_path):
    counts = [sum(labels==i) for i in range(len(cluster_top_words))]
    labels_with_words = [f"Cluster {i}: {', '.join(cluster_top_words[i])}" for i in range(len(cluster_top_words))]
    plt.figure(figsize=(8,8))
    plt.pie(counts, labels=labels_with_words, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# --------------------
# 8️⃣ Apply to questions
# --------------------
q_top_words = get_top_words(questions, q_labels)
plot_cluster_bar(q_labels, "Question Clusters", "question_clusters_bar.png")
plot_cluster_pie(q_labels, q_top_words, "Question Clusters with Top Words", "question_clusters_pie.png")

# --------------------
# 9️⃣ Apply to tables
# --------------------
t_top_words = get_top_words(tables, t_labels)
plot_cluster_bar(t_labels, "Table Clusters", "table_clusters_bar.png")
plot_cluster_pie(t_labels, t_top_words, "Table Clusters with Top Words", "table_clusters_pie.png")

# --------------------
# 10️⃣ Optional: inspect sample items
# --------------------
for cluster_id in range(n_clusters):
    cluster_questions = [q for q, lbl in zip(questions, q_labels) if lbl == cluster_id]
    print(f"\nCluster {cluster_id} sample questions ({len(cluster_questions)} total):")
    print(cluster_questions[:5])
