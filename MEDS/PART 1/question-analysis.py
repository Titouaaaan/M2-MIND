import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# same change this if needed
DATA_FILE = r"C:\Users\titouan\OneDrive\Bureau\M2-MIND\MEDS\PART 1\WikiTableQuestions-1.0.2-compact\WikiTableQuestions\data\training.tsv"

# Load questions
df = pd.read_csv(DATA_FILE, sep='\t')
questions = df['utterance'].dropna().tolist()
print(f"Loaded {len(questions)} questions")

# --- Question length stats ---
q_lengths = [len(q.split()) for q in questions]
print(f"Mean question length: {sum(q_lengths)/len(q_lengths):.2f} words")
print(f"Median question length: {pd.Series(q_lengths).median():.2f} words")

plt.figure(figsize=(7,5))
plt.hist(q_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Question Lengths")
plt.xlabel("Number of words")
plt.ylabel("Number of questions")
plt.savefig("question_length_distribution.png")
plt.show()

# --- Word frequencies / word cloud ---
stop_words = set(stopwords.words('english'))
all_words = [word.lower() for q in questions for word in q.split() if word.lower() not in stop_words]
word_freq = Counter(all_words)

# Top 20 words bar plot
top_words = word_freq.most_common(20)
words, counts = zip(*top_words)
plt.figure(figsize=(10,5))
plt.bar(words, counts, color='salmon', edgecolor='black')
plt.xticks(rotation=45)
plt.title("Top 20 Most Frequent Words in Questions")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("question_top_words.png")
plt.show()

# Word cloud
wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
plt.figure(figsize=(15,7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig("question_wordcloud.png")
plt.show()
