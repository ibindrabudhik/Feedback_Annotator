import streamlit as st
import pandas as pd
import ast
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Feedback Annotation App", layout="wide")

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        st.error("⚠️ Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
        st.stop()
    return create_client(url, key)

supabase: Client = init_supabase()

@st.cache_data
def load_data():
    df_indo = pd.read_csv("50_dataset_soal_indo_with_feedback_parsed_gpt_5nano.csv")
    df_en = pd.read_csv("50_dataset_soal_indo_with_feedback_parsed.csv")
    return df_indo, df_en

df_indo, df_en = load_data()

# Initialize session state
if 'teacher_name' not in st.session_state:
    st.session_state.teacher_name = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = "Indonesian Dataset"
if 'annotations_submitted' not in st.session_state:
    st.session_state.annotations_submitted = []

def chat_bubble(message, sender="user"):
    """
    sender = 'user'  -> student (right bubble, green)
    sender = 'ai'    -> tutor (left bubble, blue)
    """
    if sender == "ai":
        st.markdown(
            f"""
            <div style="background-color: #e6f2ff; color: black; padding:15px; border-radius: 15px; 
                        max-width: 75%; margin: 10px 10px 10px 0; display: inline-block;">
                <strong>🤖 Tutor:</strong><br>{message}
            </div>
            <br style="clear: both;">
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="text-align: right; margin-bottom: 10px;">
                <div style="background-color: #d9ffd9; color: black; padding:15px; border-radius: 15px; 
                            max-width: 75%; margin-left: auto; display: inline-block; text-align: left;">
                    <strong>👤 Student:</strong><br>{message}
                </div>
            </div>
            <br style="clear: both;">
            """,
            unsafe_allow_html=True
        )

def save_annotation(teacher_name, dataset_name, row_data, annotations):
    """Save annotation to Supabase"""
    try:
        data = {
            "teacher_name": teacher_name,
            "dataset_name": dataset_name,
            "row_index": int(row_data.get("No", st.session_state.current_index)),
            "problem": str(row_data.get("Soal", row_data.get("question", ""))),
            "student_answer": str(row_data.get("Jawaban_Salah", row_data.get("student_error", ""))),
            "correct_answer": str(row_data.get("Jawaban", row_data.get("correct_answer", ""))),
            "generated_feedback": str(row_data.get("Generated_Feedback", "")),
            "relevancy": annotations["relevancy"],
            "accuracy": annotations["accuracy"],
            "motivation": annotations["motivation"],
            "demotivation": annotations["demotivation"],
            "guidance": annotations["guidance"],
            "tone_style": annotations["tone_style"],
            "timestamp": datetime.now().isoformat()
        }
        
        response = supabase.table("annotations").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        return False

def teacher_login():
    """Teacher name entry screen"""
    st.title("📝 Feedback Annotation System")
    st.markdown("### Welcome, Teacher!")
    st.markdown("Please enter your name to begin annotating student feedback.")
    
    with st.form("teacher_login_form"):
        teacher_name = st.text_input("Your Name:", placeholder="e.g., Dr. Smith")
        dataset_choice = st.radio("Select Dataset to Annotate:", 
                                  ["Indonesian Dataset (GPT-5-nano)", "English Dataset (GPT-4o)"])
        submit = st.form_submit_button("Start Annotation")
        
        if submit and teacher_name:
            st.session_state.teacher_name = teacher_name.strip()
            st.session_state.selected_dataset = dataset_choice
            st.session_state.current_index = 0
            st.rerun()
        elif submit:
            st.error("Please enter your name.")

def annotation_interface():
    """Main annotation interface"""
    # Get current dataset
    if "Indonesian" in st.session_state.selected_dataset:
        df = df_indo
        dataset_name = "Indonesian Dataset (GPT-5-nano)"
    else:
        df = df_en
        dataset_name = "English Dataset (GPT-4o)"
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👨‍🏫 Teacher: {st.session_state.teacher_name}")
        st.markdown(f"**Dataset:** {dataset_name}")
        st.markdown(f"**Progress:** {len(st.session_state.annotations_submitted)}/{len(df)} annotated")
        st.progress(len(st.session_state.annotations_submitted) / len(df))
        
        if st.button("🚪 Logout"):
            st.session_state.teacher_name = None
            st.session_state.current_index = 0
            st.session_state.annotations_submitted = []
            st.rerun()
    
    # Check if all done
    if st.session_state.current_index >= len(df):
        st.success("🎉 You have completed all annotations!")
        st.balloons()
        if st.button("Start Over"):
            st.session_state.current_index = 0
            st.session_state.annotations_submitted = []
            st.rerun()
        return
    
    # Get current row
    current_idx = st.session_state.current_index
    row = df.iloc[current_idx]
    
    st.title(f"Problem Set #{current_idx + 1}")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Student Profile & Problem Details
        st.markdown("### 📋 Student Profile & Problem Details")
        
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px; margin-bottom: 15px; color: black;">
            <strong>Problem Number:</strong> {row.get('No', current_idx + 1)}<br>
            <strong>Knowledge Level (SPK):</strong> {row.get('SPK', 'N/A')}<br>
            <strong>Mistake Level (SAL):</strong> {row.get('SAL', 'N/A')}<br>
            <strong>Feedback Type:</strong> {row.get('Final_Feedback_Type', 'N/A')}<br>
        </div>
        """, unsafe_allow_html=True)
        
        # Problem Statement
        problem = row.get("Soal", row.get("question", ""))
        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; color: black;">
            <strong>📝 Problem:</strong><br>{problem}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Correct Answer
        correct_answer = row.get("Jawaban", row.get("correct_answer", ""))
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; color: black;">
            <strong>✅ Correct Answer:</strong> {correct_answer}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Chat-style conversation
        st.markdown("### 💬 Student-Tutor Conversation")
        
        # Parse feedback
        feedback_text = ""
        if "dict_generated_feedback" in row and pd.notna(row["dict_generated_feedback"]):
            try:
                parsed = ast.literal_eval(row["dict_generated_feedback"])
                feedback_text = parsed.get("Feedback", "")
            except:
                feedback_text = str(row.get("Generated_Feedback", ""))
        else:
            feedback_text = str(row.get("Generated_Feedback", ""))
        
        student_answer = row.get("Jawaban_Salah", row.get("student_error", ""))
        
        # Display chat
        st.markdown("<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>", 
                   unsafe_allow_html=True)
        chat_bubble(student_answer, sender="user")
        chat_bubble(feedback_text, sender="ai")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Annotation Form
    st.markdown("---")
    st.markdown("### 📊 Annotation Form")
    st.markdown("Please rate the tutor's feedback on the following criteria:")
    
    with st.form("annotation_form"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            relevancy = st.radio(
                "1. Relevancy with Formative Feedback",
                options=[1, 2, 3, 4],
                format_func=lambda x: f"{x} - {'Very Low' if x==1 else 'Low' if x==2 else 'High' if x==3 else 'Very High'}",
                help="How relevant is the feedback to formative assessment principles?"
            )
            
            accuracy = st.radio(
                "2. Accuracy",
                options=[0, 1],
                format_func=lambda x: "Incorrect" if x == 0 else "Correct",
                help="Is the feedback mathematically/factually accurate?"
            )
        
        with col_b:
            motivation = st.radio(
                "3. Motivation",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} - {'Low' if x==1 else 'Medium' if x==2 else 'High'}",
                help="How motivating is the feedback for the student?"
            )
            
            demotivation = st.radio(
                "4. Demotivation",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} - {'Low' if x==1 else 'Medium' if x==2 else 'High'}",
                help="How demotivating is the feedback? (Lower is better)"
            )
        
        with col_c:
            guidance = st.radio(
                "5. Guidance",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} - {'Low' if x==1 else 'Medium' if x==2 else 'High'}",
                help="How well does the feedback guide the student?"
            )
            
            tone_style = st.radio(
                "6. Tone and Style",
                options=[1, 2, 3, 4],
                format_func=lambda x: f"{x} - {'Poor' if x==1 else 'Fair' if x==2 else 'Good' if x==3 else 'Excellent'}",
                help="How appropriate is the tone and style of the feedback?"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_submit, col_skip = st.columns([1, 4])
        with col_submit:
            submit_button = st.form_submit_button("✅ Submit & Next", use_container_width=True)
        
        if submit_button:
            annotations = {
                "relevancy": relevancy,
                "accuracy": accuracy,
                "motivation": motivation,
                "demotivation": demotivation,
                "guidance": guidance,
                "tone_style": tone_style
            }
            
            # Save to database
            if save_annotation(st.session_state.teacher_name, dataset_name, row, annotations):
                st.session_state.annotations_submitted.append(current_idx)
                st.session_state.current_index += 1
                st.success("✅ Annotation saved successfully!")
                st.rerun()

# Main app logic
if st.session_state.teacher_name is None:
    teacher_login()
else:
    annotation_interface()
