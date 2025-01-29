import json
import spacy
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# بارگذاری مدل زبان فارسی spacy
nlp = spacy.blank("fa")

# لیست استاپ‌وردهای فارسی
stop_words = set([
    "و", "در", "به", "از", "که", "این", "را", "با", "برای", "است", "آن",
    "تا", "می", "بر", "شد", "ای", "یا", "هم", "یک", "دارد", "هر", "چند",
    "بود", "دیگر", "اما", "اگر", "نه", "نیز", "بین"
])

# تابع برای پیش‌پردازش متن
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.text not in stop_words and not token.is_punct]
    return " ".join(tokens)

# تابع ساخت و ذخیره فایل JSONL
def create_fresh_jsonl(filename):
    articles = []
    
    # لیست مقالات (نمونه)
    raw_texts = [
        "متن مقاله اول نمونه",
        "اصغر سیبیل",
        "بی ادبی بیشعروی",
        "کودکان بدبخت",
        "متن مقاله سوم نمونه",

    ]
    
    sample_questions = [
        ["سوال ۱ مقاله ۱", "جواب ۱ مقاله ۱"],
        ["سوال ۱ مقاله ۲", "جواب ۱ مقاله ۲"],
        ["سوال ۱ مقاله ۳", "جواب ۱ مقاله ۳"],
        ["سوال ۱ مقاله ۴", "جواب ۱ مقاله ۴"],
        ["سوال ۱ مقاله ۵", "جواب ۱ مقاله ۵"]
    ]
    
    for i, raw_text in enumerate(raw_texts):
        cleaned_text = preprocess_text(raw_text)
        article = {
            "raw_text": raw_text,
            "raw_text_prp": cleaned_text,
            "question1": sample_questions[i][0],
            "answer1": sample_questions[i][1]
        }
        articles.append(article)
    
    # ذخیره در فایل JSONL
    with open(filename, "w", encoding="utf-8") as file:
        for article in articles:
            file.write(json.dumps(article, ensure_ascii=False) + "\n")
    
    print(f"فایل {filename} با موفقیت ایجاد شد.")
    return articles

# تابع ساخت ماتریس TF-IDF و محاسبه شباهت کسینوسی
def compute_similarity(articles):
    documents = [article["raw_text_prp"] for article in articles]
    
    # حذف کلمات پرتکرار و کم‌تکرار
    all_words = " ".join(documents).split()
    word_counts = Counter(all_words)
    filtered_documents = []
    
    for doc in documents:
        filtered_tokens = [word for word in doc.split() if word_counts[word] > 1]
        filtered_documents.append(" ".join(filtered_tokens))
    
    # تبدیل متون به بردارهای عددی با TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_documents)
    
    # محاسبه شباهت کسینوسی
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    
    # پیدا کردن بیشترین شباهت به مقاله اول
    most_similar_idx = np.argmax(similarities)
    most_similar_score = similarities[0][most_similar_idx]
    
    print(f"مقاله‌ای که بیشترین شباهت را به مقاله اول دارد: مقاله {most_similar_idx + 2} با امتیاز شباهت {most_similar_score:.3f}")

# اجرای کد
articles = create_fresh_jsonl("articles.jsonl")
compute_similarity(articles)
