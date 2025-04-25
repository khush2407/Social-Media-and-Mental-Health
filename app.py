import warnings
warnings.filterwarnings("ignore")

import streamlit as st
# ─── Must be the first Streamlit call ───────────────────────────────────────
st.set_page_config(page_title="Depression Risk Predictor", layout="wide")

import pandas as pd
import pickle

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .section { padding: 1rem; border-radius: 0.75rem; margin-bottom: 1rem; }
      .form-section  { background: #f0f2f6; }
      .info-section  { background: #e2f0fb; }
      .tips-section  { background: #e8f5e9; }
      @media (max-width: 768px) {
        .stColumns { flex-direction: column !important; }
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Load & cache the model ─────────────────────────────────────────────────
@st.cache_resource
def load_model(path="ML_Social_Media_Mental_Health.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ─── Load & cache the survey data ────────────────────────────────────────────
@st.cache_data
def load_survey(path="smmh.csv"):
    return pd.read_csv(path)

df = load_survey()

# ─── Sidebar navigation ─────────────────────────────────────────────────────
page = st.sidebar.selectbox("Navigate to", ["Predict", "Infographic", "Tips"])

# ─── Predict page ───────────────────────────────────────────────────────────
if page == "Predict":
    st.markdown(
    "<h1 style='text-align: center;'> Depression Risk Prediction</h1>",
    unsafe_allow_html=True
    )

    

    st.markdown(
        """
        <div style="text-align: center;">
          <img
            src="https://www.priorygroup.com/media/zfpbagxe/social-media-mental-health-impact-tips-infographic.jpg"
            width="600"
            height="450"
            style="object-fit: cover;"
          />
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )



    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input(
            "Age", 10, 100, 25,
            help="Age can influence mental health risk; different life stages face different stressors."
        )
        gender = c2.selectbox(
            "Gender", ["Male", "Female", "Other"],
            help="Gender differences can affect how depression manifests and is reported."
        )
        rel = st.selectbox(
            "Relationship Status", ["Single", "Married", "Other"],
            help="Social support from relationships can buffer against depressive symptoms."
        )
        occ = st.selectbox(
            "Occupation", ["School Student", "University Student", "Employed", "Other"],
            help="Daily roles and responsibilities influence stress and mental load."
        )
        use_sm = st.selectbox(
            "Use social media?", ["Yes", "No"],
            help="Active social-media use has been linked to both positive and negative mental-health outcomes."
        )
        avg_time = st.selectbox(
            "Avg. daily SM time", ["<1h", "1-2h", "2-3h", "3-4h", ">4h"],
            help="Longer daily usage often correlates with greater risk of anxiety or mood disturbances."
        )

        st.markdown("#### Rate these (1 = low … 5 = high)")
        q9 = st.slider("Purpose-free SM use", 1, 5, 3, help="Mindless scrolling can increase rumination.")
        q10 = st.slider("Distracted when busy", 1, 5, 3, help="Frequent distraction indicates difficulty focusing.")
        q11 = st.slider("Restless without SM", 1, 5, 3, help="Restlessness may reflect dependency or anxiety.")
        q12 = st.slider("Ease of distraction", 1, 5, 3, help="High distractibility links to anxiety/depression.")
        q13 = st.slider("Bothered by worries", 1, 5, 3, help="Excessive worry is a core feature of mood disorders.")
        q14 = st.slider("Difficulty concentrating", 1, 5, 3, help="Concentration problems are common in depression.")
        q16 = st.slider("Feelings about comparisons", 1, 5, 3, help="Negative comparisons lower self-esteem.")
        q17 = st.slider("Seeking validation online", 1, 5, 3, help="Relying on online feedback can harm well-being.")
        q19 = st.slider("Interest fluctuation", 1, 5, 3, help="Fluctuating interest may signal anhedonia.")
        q20 = st.slider("Sleep issues frequency", 1, 5, 3, help="Sleep disturbances are both symptom and risk factor.")

        submitted = st.form_submit_button("Submit")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        inputs = {
            "1. What is your age?": age,
            "2. Gender": gender,
            "3. Relationship Status": rel,
            "4. Occupation Status": occ,
            "6. Do you use social media?": use_sm,
            "8. What is the average time you spend on social media every day?": avg_time,
            "9. How often do you find yourself using Social media without a specific purpose?": q9,
            "10. How often do you get distracted by Social media when you are busy doing something?": q10,
            "11. Do you feel restless if you haven't used Social media in a while?": q11,
            "12. On a scale of 1 to 5, how easily distracted are you?": q12,
            "13. On a scale of 1 to 5, how much are you bothered by worries?": q13,
            "14. Do you find it difficult to concentrate on things?": q14,
            "16. Following the previous question, how do you feel about these comparisons, generally speaking?": q16,
            "17. How often do you look to seek validation from features of social media?": q17,
            "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?": q19,
            "20. On a scale of 1 to 5, how often do you face issues regarding sleep?": q20
        }
        df_in = pd.DataFrame([inputs])

        prob  = model.predict_proba(df_in)[:, 1][0]
        label = model.predict(df_in)[0]

        st.subheader("Prediction Results")
        st.write(f"**High-Risk Probability:** {prob:.1%}")
        if label == 1:
            st.error("⚠️ You are at **high risk** of depressive symptoms.")
        else:
            st.success("✅ You are at **low risk** of depressive symptoms.")

# ─── Infographic page (Text-only) ─────────────────────────────────────────────
elif page == "Infographic":
    st.header("📊 Infographic: Expert Reading")

    st.subheader("📰 Like It or Not, Social Media’s Affecting Your Mental Health")
    st.markdown("""
- **Dopamine Loop & Addiction**  
  Social media taps into your brain’s reward center with unpredictable “likes,” driving repeated check-ins and mood swings.

- **Scope of the Issue**  
  About 69% of U.S. adults and 81% of teens use social media—linking heavy use to anxiety, depression, insomnia, and even physical symptoms like headaches.

- **Comparison & FOMO**  
  Even without visible “likes,” users compare follower counts and comments. Fear of missing out fuels compulsive scrolling and erodes self-esteem.

- **Vulnerability of Teens**  
  Young people, especially girls, experience relational aggression and social exclusion online. Early exposure magnifies anxiety when impulse control is still developing.

- **Behavioral Interventions**  
  Capping social-media use to 10 minutes per platform daily has been shown to significantly reduce loneliness and depressive symptoms in undergraduates over three weeks.

**Take action:**  
Self-monitor your mood before and after social-media sessions, then set clear boundaries (for example, no apps after 8 PM) to regain control.
    """)



# ─── Tips page ───────────────────────────────────────────────────────────────
else:
    st.header("💡 5 Steps to Boost Your Well-Being")
    st.markdown('<div class="section tips-section">', unsafe_allow_html=True)
    st.markdown("""
    1. **Maintain a consistent sleep schedule** (7–9 hrs/night)  
    2. **Schedule social-media breaks** (phone-free blocks)  
    3. **Practice mindfulness** or meditation (5–10 min/day)  
    4. **Engage in physical activity** (e.g., walking, yoga) ≥ 3×/week  
    5. **Reach out** to a trusted friend or mental-health professional  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2025 Depression Risk Predictor — not a substitute for medical advice.")
st.markdown(
    """
    <style>
      footer { visibility: hidden; }
      .stApp { padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)