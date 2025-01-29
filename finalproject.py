import json
import spacy
import numpy as np
from hazm import *
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# بارگذاری مدل‌های مورد نیاز
nlp = spacy.blank("fa")
normalizer = Normalizer()
lemmatizer = Lemmatizer()
stemmer = Stemmer()
tokenizer = WordTokenizer()

# لیست استاپ‌وردهای فارسی
stop_words = set([
    "و", "در", "به", "از", "که", "را", "با", "این", "است", "برای",
    "تا", "می", "بر"
])

def preprocess_text(text):
    """
    پیش‌پردازش متن فقط برای متن اصلی
    """
    # نرمال‌سازی متن
    text = normalizer.normalize(text)
    
    # تبدیل به توکن
    words = tokenizer.tokenize(text)
    
    # حذف استاپ‌وردها و کلمات کوتاه
    words = [word for word in words if word not in stop_words and len(word) > 1]
    
    # ریشه‌یابی
    words = [stemmer.stem(word) for word in words]
    
    # چاپ تعداد کلمات برای اشکال‌زدایی
    print(f"تعداد کلمات متن پس از پیش‌پردازش: {len(words)}")
    
    return " ".join(words)

def create_fresh_jsonl(filename):
    articles = [
        {
            "raw_text": """حدیث لوح حضرت فاطمه زهرا(س)، از احادیث منقول از پیامبر اسالم(ص) برای اثبات امامت ائمه دوازده‌گانه شیعه است. این حدیث با توجه به مقبولیت سند و فراوانی نقل آن در منابع مختلف، جایگاه ویژه‌ای در اثبات امامت و خلافت ائمه شیعه دارد.""",
            "raw_text_prp": "",
            "question1": "حدیث لوح چیست؟",
            "answer1": "حدیثی منقول از پیامبر اسالم(ص) برای اثبات امامت ائمه دوازده‌گانه شیعه است."
        },
        {
            "raw_text": """تاریخ اسلام شامل وقایع مهم صدر اسلام و زندگی پیامبر اکرم(ص) است. 
                          این تاریخ از بعثت پیامبر شروع شده و شامل وقایع مهم مکه و مدینه 
                          می‌باشد. هجرت پیامبر از مکه به مدینه نقطه عطف این تاریخ است.""",
            "raw_text_prp": "",
            "question1": "تاریخ اسلام از چه زمانی شروع می‌شود؟",
            "answer1": "از زمان بعثت پیامبر اکرم(ص)"
        },
         {
            "raw_text": """تاریخ اسلام شامل وقایع مهم صدر اسلام و زندگی پیامبر اکرم(ص) است. 
                          این تاریخ از بعثت پیامبر شروع شده و شامل وقایع مهم مکه و مدینه 
                          می‌باشد. هجرت پیامبر از مکه به مدینه نقطه عطف این تاریخ است.""",
            "raw_text_prp": "",
            "question1": "تاریخ اسلام از چه زمانی شروع می‌شود؟",
            "answer1": "از زمان بعثت پیامبر اکرم(ص)"
        },
        {
            "raw_text": """حدیث لوح حضرت فاطمه زهرا(س)، از احادیث منقول از پیامبر اسالم(ص) 
                          برای اثبات امامت ائمه دوازده‌گانه شیعه است. این حدیث با توجه به 
                          مقبولیت سند و فراوانی نقل آن در منابع مختلف، جایگاه ویژه‌ای در 
                          اثبات امامت و خلافت ائمه شیعه دارد.""",
            "raw_text_prp": "",
            "question1": "حدیث لوح چیست؟",
            "answer1": "حدیثی منقول از پیامبر اسالم(ص) برای اثبات امامت ائمه دوازده‌گانه شیعه است."
        },
    ]
    
    # فقط پیش‌پردازش متن اصلی
    for article in articles:
        article["raw_text_prp"] = preprocess_text(article["raw_text"])
    
    # ذخیره در فایل JSONL
    with open(filename, "w", encoding="utf-8") as file:
        for article in articles:
            file.write(json.dumps(article, ensure_ascii=False) + "\n")
    
    print(f"فایل {filename} با موفقیت ایجاد شد.")
    return articles

def compute_similarity(articles):
    # فقط استفاده از متن‌های اصلی پیش‌پردازش شده
    documents = [article["raw_text_prp"] for article in articles]
    
    if not all(documents):
        print("خطا: برخی از متن‌ها خالی هستند!")
        return
    
    # حذف کلمات پرتکرار و کم‌تکرار
    all_words = " ".join(documents).split()
    word_counts = Counter(all_words)
    
    # تنظیم آستانه‌های مناسب
    min_freq = 1
    max_freq = len(documents) * 0.95
    
    # فیلتر کردن کلمات
    filtered_documents = []
    for doc in documents:
        filtered_tokens = [word for word in doc.split() 
                         if min_freq <= word_counts[word] <= max_freq]
        if filtered_tokens:
            filtered_documents.append(" ".join(filtered_tokens))
        else:
            print(f"هشدار: یکی از متن‌ها پس از فیلتر خالی شد!")
            filtered_documents.append(doc)
    
    # محاسبه TF-IDF
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)
    
    try:
        # تبدیل متن‌ها به بردار
        tfidf_matrix = vectorizer.fit_transform(filtered_documents)
        
        # محاسبه شباهت کسینوسی با متن اول
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # نمایش نتایج
        print("\nنتایج تحلیل شباهت متن‌های اصلی:")
        print("=" * 50)
        print("تعداد کلمات کلیدی استخراج شده:", len(vectorizer.vocabulary_))
        print("\nشباهت متن اول با سایر متن‌ها:")
        for idx, score in enumerate(similarities[0], start=2):
            print(f"شباهت با متن {idx}: {score:.3f}")
        
        # پیدا کردن شبیه‌ترین متن
        most_similar_idx = np.argmax(similarities[0])
        most_similar_score = similarities[0][most_similar_idx]
        print(f"\nشبیه‌ترین متن: متن شماره {most_similar_idx + 2} با امتیاز شباهت {most_similar_score:.3f}")
        
    except ValueError as e:
        print(f"خطا در پردازش متن: {e}")
        print("لطفاً متن‌های ورودی را بررسی کنید.")

# اجرای کد
if __name__ == "__main__":
    articles = create_fresh_jsonl("articles.jsonl")
    compute_similarity(articles)