import os
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from google import generativeai as genai
from PyPDF2 import PdfReader

# 🌿 1. Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 🌿 2. PDF verisini oku
reader = PdfReader("kariyeralanlari.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

clean_text = (
    text.replace("\n", " ")
    .replace("  ", " ")
    .strip()
)

# 🌿 3. Text'i ChromaDB'ye yükle
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="kariyer_verisi")

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = clean_text.split("Kaynak:")
for i, chunk in enumerate(chunks):
    if chunk.strip():
        embedding = model.encode([chunk])
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            metadatas=[{"role": f"Alan {i}"}]
        )

# 🌿 4. Sorgu için fonksiyon
def retrieve_similar_chunks(query, top_k=3):
    query_emb = model.encode([query])
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    return results['documents'][0]

def generate_response(user_input):
    related_chunks = retrieve_similar_chunks(user_input)
    context = "\n\n".join(related_chunks)

    prompt = f"""
    Sen deneyimli bir kariyer danışmanısın.
    Aşağıda endüstri mühendisliği mezunları için hazırlanmış kariyer alanlarıyla ilgili bilgiler yer alıyor:

    {context}

    Kullanıcının kendini tanıttığı bilgiler:
    {user_input}

    Bu bilgiler ışığında kullanıcıya uygun 1-2 kariyer alanı öner.
    Her bir alan için neden uygun olduğunu açıkla.
    Ardından bu alanda gelişmek için 3 pratik öneri sun.
    """

    model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model_gemini.generate_content(prompt)
    return response.text.strip().replace("**", "").replace("###", "")

# 🌿 5. Gradio Arayüzü
def chatbot_interface(user_input):
    return generate_response(user_input)

interface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(
        label="🎓 Kendini Tanıt",
        placeholder="Kendini birkaç cümleyle tanıt...",
        lines=3,
        max_lines=10
    ),
    outputs=gr.Textbox(
        label="💡 Kariyer Önerisi",
        lines=20,
        max_lines=40
    ),
    title="Endüstri Mühendisliği Kariyer Danışmanı 🤖",
    description="Kendini tanıt, bot sana en uygun kariyer alanlarını önersin 🌱",
    theme="soft",
)

if __name__ == "__main__":
    interface.launch(share=True)
