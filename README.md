# 🎓 Endüstri Mühendisliği Kariyer Danışmanı

Generative AI Bootcamp kapsamında hazırlanmış RAG (Retrieval-Augmented Generation) tabanlı bir Türkçe chatbot projesi.
Bu proje, endüstri mühendisliği öğrencilerinin kendi ilgi alanları, becerileri ve kişisel özelliklerine göre en uygun kariyer alanlarını keşfetmelerini sağlar.
---

## 📋 Proje Hakkında

Bu chatbot, kullanıcının birkaç cümleyle kendisini tanıttığı kısa bir metni 
(örneğin: *“Analitik düşünen, veriyle çalışmayı seven ve takım çalışmasına yatkınım.”*) 
girdi olarak alır.  
Metni analiz ederek veri tabanındaki kariyer profilleriyle eşleştirir ve kullanıcıya en uygun alanları önerir.  
Ayrıca her öneri için **neden uygun olduğunu açıklar** ve **kişisel gelişim önerileri** sunar.

---
## 🛠️ Kullanılan Teknolojiler

- **LangChain** – RAG pipeline framework  
- **ChromaDB** – Vektör veritabanı  
- **Sentence Transformers** – Embedding modeli  
- **Google Gemini 2.5 Flash** – Metin üretimi  
- **Gradio** – Web arayüzü  
- **PyPDF2** ve **dotenv** – Dosya & API yönetimi  
