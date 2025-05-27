import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer
import google.generativeai as genai

st.title("Personalized Exam Preparation AI")
st.sidebar.title("AI Study Buddy")
tab =st.sidebar.selectbox("Choose Feature", [
    "Summarize Chapter",
    "Generate Questions",
    "Ask a Question",
    "Create MCQ's",
    "Explain in Urdu"
])

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",device=-1)

@st.cache_resource
def load_mcq_generator():
    return pipeline("text2text-generation", model="valhalla/t5-base-qg-hl",device=-1)

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)

@st.cache_resource
def load_flashcard_generator():
    return pipeline("text-generation", model="declare-lab/flan-alpaca-base", tokenizer="declare-lab/flan-alpaca-base",device=-1)

@st.cache_resource
def load_translator():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
    return tokenizer, model

GEMINI_API_KEY ="AIzaSyCv86uC98jkzdzO6eCR4Tjav1AcGxIT2gQ" 

if tab =="Summarize Chapter":
    chapter = st.text_area("Paste chapter content here",height=200)
    if st.button("Summarize"):
        summarizer = load_summarizer()
        summary = summarizer(chapter,max_length=100,min_length=30,do_sample=False)[0]['summary_text']
        st.subheader("Summary")
        st.write(summary)

elif tab =="Generate Questions":
    content = st.text_area("Paste content for Question generation", height=200)
    num_mcqs = st.slider("Select number of questions", 1, 4)

    if st.button("Generate Questions"):
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        highlighted = ""
        for i in range(min(num_mcqs, len(sentences))):
            highlighted += f"<hl> {sentences[i]} <hl> "
        prompt = f"generate questions: {highlighted} context: {content}"

        generator = load_mcq_generator()
        try:
            results = generator(prompt,max_length=64,num_return_sequences=num_mcqs)
        except Exception as e:
            st.error(f"Model generation failed: {e}")
            st.stop()

        st.subheader("Generated Questions")
        for i, res in enumerate(results):
            st.markdown(f"**Q{i+1}: {res['generated_text']}**")

elif tab =="Ask a Question":
    context =st.text_area("Paste study material (context)",height=200)
    question =st.text_input("Your question")
    if st.button("Get Answer"):
        qa_pipeline =load_qa_pipeline()
        answer =qa_pipeline(question=question, context=context)
        st.subheader("Answer")
        st.write(answer['answer'])

elif tab =="Create MCQ's":
    st.subheader("Create Multiple Choice Questions (MCQs)")
    content =st.text_area("Paste content for MCQ creation", height=200)
    num_mcqs =st.slider("Select number of MCQs", 1, 10)
    if st.button("Generate MCQs "):
        prompt =(
            f"Generate {num_mcqs} multiple choice questions (MCQs) with 4 options (A, B, C, D), the correct answer, and a short explanation for each, based on the following content:\n\n"
            f"{content}\n\n"
            "Format:\n"
            "Q: <question>\n"
            "A. <option>\n"
            "B. <option>\n"
            "C. <option>\n"
            "D. <option>\n"
            "Answer: <correct option letter>\n"
            "Explanation: <short explanation>\n"
        )
        with st.spinner("Generating MCQs..."):
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
                mcq_text = response.text.strip()
                st.subheader("Generated MCQs")
                st.text_area("MCQs", mcq_text, height=300)
            except Exception as e:
                st.error(f"Gemini API call failed: {e}")

elif tab =="Explain in Urdu":
    text =st.text_area("Enter text to explain in Urdu", height=200)
    if st.button("Explain to Urdu"):
        prompt =(
            f"Translate the following text to Urdu and explain it in simple terms for a student:\n\n"
            f"{text}\n\n"
            "Output only in Urdu."
        )
        with st.spinner("Translating and explaining..."):
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                model =genai.GenerativeModel("gemini-2.0-flash")
                response =model.generate_content(prompt)
                urdu_explanation =response.text.strip()
                st.subheader("Explanation in Urdu")
                st.write(urdu_explanation)
            except Exception as e:
                st.error(f"Gemini API call failed: {e}")
