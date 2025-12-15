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
    # Dataset file paths - update these when you have the actual files
    datasets = {
        "RAG4O": "combined_data_generated_feedback_rag_gpt4o.csv",  # RAG WITH GPT4O
        "RAG5N": "combined_data_generated_feedback_ragGPT-5-nano.csv",  # RAG WITH GPT5NANO
        "4O": "combined_data_generated_feedback_GPT4o.csv",  # GPT4O
        "5N": "combined_data_generated_feedback_GPT-5-nano.csv"  # GPT5NANO
    }
    
    loaded_datasets = {}
    for name, path in datasets.items():
        try:
            loaded_datasets[name] = pd.read_csv(path)
        except FileNotFoundError:
            # Create dummy dataframe for testing if file doesn't exist
            loaded_datasets[name] = pd.DataFrame({
                'No': [1, 2, 3],
                'Soal': [f'Sample problem {i} for {name}' for i in range(1, 4)],
                'Jawaban': ['17', '25', '30'],
                'Jawaban_Salah': ['15', '20', '28'],
                'SPK': ['High', 'Medium', 'Low'],
                'SAL': ['High', 'Medium', 'Low'],
                'Final_Feedback_Type': ['Try again', 'Hint', 'Explanation'],
                'Generated_Feedback': [f'Sample feedback {i}' for i in range(1, 4)],
                'dict_generated_feedback': ["{'Feedback': 'Sample feedback text'}" for _ in range(3)]
            })
    
    return loaded_datasets

datasets = load_data()

def get_annotated_rows(teacher_name, dataset_name):
    """Fetch list of row indices already annotated by this teacher for this dataset"""
    try:
        response = supabase.table("annotations").select("row_index").eq("teacher_name", teacher_name).eq("dataset_name", dataset_name).execute()
        annotated_indices = [row['row_index'] for row in response.data]
        return set(annotated_indices)
    except Exception as e:
        st.warning(f"Could not fetch previous annotations: {str(e)}")
        return set()

# Initialize session state
if 'teacher_name' not in st.session_state:
    st.session_state.teacher_name = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = "RAG4O"
if 'annotations_submitted' not in st.session_state:
    st.session_state.annotations_submitted = set()
if 'unannotated_rows' not in st.session_state:
    st.session_state.unannotated_rows = []

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
        # Get the actual row index from the dataframe
        row_index = int(row_data.get("No", st.session_state.current_index))
        print(row_data.get("Soal_x",""))
        data = {
            "teacher_name": teacher_name,
            "dataset_name": dataset_name,
            "row_index": row_index,
            "problem": str(row_data.get("Soal_x", row_data.get("Soal_y", row_data.get("Soal", "")))),
            "student_answer": str(row_data.get("Jawaban_Salah", row_data.get("Student_Answer", ""))),
            "correct_answer": str(row_data.get("Jawaban", row_data.get("correct_answer", ""))),
            "generated_feedback": str(row_data.get("Generated_Feedback", "")),
            "relevancy": annotations["relevancy"],
            "accuracy": annotations["accuracy"],
            "motivation": annotations["motivation"],
            "demotivation": annotations["demotivation"],
            "guidance": annotations["guidance"],
            "tone_style": annotations["tone_style"],
            "teacher_comments": annotations.get("teacher_comments"),
            "timestamp": datetime.now().isoformat()
        }
        
        response = supabase.table("annotations").insert(data).execute()
        return True
    except Exception as e:
        error_msg = str(e)
        # Check if it's a duplicate key error
        if "duplicate key" in error_msg.lower() or "23505" in error_msg:
            st.warning("⚠️ This question was already annotated. Skipping to next question.")
            # Treat as success so we can move to next question
            return True
        else:
            st.error(f"Error saving to database: {error_msg}")
            return False

def teacher_login():
    """Teacher name entry screen"""
    st.title("📝 Feedback Annotation System")
    st.markdown("### Welcome, Teacher!")
    st.markdown("Please enter your name to begin annotating student feedback.")
    
    with st.form("teacher_login_form"):
        teacher_name = st.text_input("Your Name:", placeholder="e.g., Dr. Smith")
        dataset_choice = st.radio("Select Dataset to Annotate:", 
                                  ["RAG4O", "RAG5N", "4O", "5N"],
                                  help="RAG4O: RAG + GPT-4o | RAG5N: RAG + GPT-o1 | 4O: GPT-4o | 5N: GPT-o1")
        submit = st.form_submit_button("Start Annotation")
        
        if submit and teacher_name:
            st.session_state.teacher_name = teacher_name.strip()
            st.session_state.selected_dataset = dataset_choice
            st.session_state.current_index = 0
            
            # Fetch already-annotated questions for this teacher and dataset
            annotated = get_annotated_rows(teacher_name.strip(), dataset_choice)
            st.session_state.annotations_submitted = annotated
            
            # Get list of unannotated rows
            df = datasets[dataset_choice]
            # Use 'No' column if available, otherwise use index
            if 'No' in df.columns:
                all_row_numbers = set(df['No'].values)
            else:
                all_row_numbers = set(range(len(df)))
            
            unannotated = sorted(list(all_row_numbers - annotated))
            st.session_state.unannotated_rows = unannotated
            
            if len(unannotated) > 0:
                st.session_state.current_index = unannotated[0]
            else:
                st.session_state.current_index = 0
            
            st.rerun()
        elif submit:
            st.error("Please enter your name.")

def annotation_interface():
    """Main annotation interface"""
    # Get current dataset
    dataset_name = st.session_state.selected_dataset
    df = datasets[dataset_name]
    
    # Get unannotated rows
    if not st.session_state.unannotated_rows:
        all_rows = set(range(len(df)))
        unannotated = sorted(list(all_rows - st.session_state.annotations_submitted))
        st.session_state.unannotated_rows = unannotated
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👨‍🏫 Teacher: {st.session_state.teacher_name}")
        
        st.markdown("---")
        st.markdown("### 📊 Switch Dataset")
        
        # Dataset selector
        new_dataset = st.selectbox(
            "Select Dataset:",
            ["RAG4O", "RAG5N", "4O", "5N"],
            index=["RAG4O", "RAG5N", "4O", "5N"].index(st.session_state.selected_dataset),
            help="RAG4O: RAG + GPT-4o | RAG5N: RAG + GPT-o1 | 4O: GPT-4o | 5N: GPT-o1"
        )
        
        # Check if dataset changed
        if new_dataset != st.session_state.selected_dataset:
            st.session_state.selected_dataset = new_dataset
            
            # Fetch annotations for new dataset
            annotated = get_annotated_rows(st.session_state.teacher_name, new_dataset)
            st.session_state.annotations_submitted = annotated
            
            # Get list of unannotated rows for new dataset
            new_df = datasets[new_dataset]
            if 'No' in new_df.columns:
                all_row_numbers = set(new_df['No'].values)
            else:
                all_row_numbers = set(range(len(new_df)))
            
            unannotated = sorted(list(all_row_numbers - annotated))
            st.session_state.unannotated_rows = unannotated
            
            if len(unannotated) > 0:
                st.session_state.current_index = unannotated[0]
            
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"**Current Dataset:** {dataset_name}")
        total_annotated = len(st.session_state.annotations_submitted)
        total_questions = len(df)
        remaining = len(st.session_state.unannotated_rows)
        st.markdown(f"**Progress:** {total_annotated}/{total_questions} annotated")
        st.markdown(f"**Remaining:** {remaining} questions")
        st.progress(total_annotated / total_questions if total_questions > 0 else 0)
        
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.teacher_name = None
            st.session_state.current_index = 0
            st.session_state.annotations_submitted = set()
            st.session_state.unannotated_rows = []
            st.rerun()
    
    # Check if all done
    if len(st.session_state.unannotated_rows) == 0:
        st.success("🎉 You have completed all annotations for this dataset!")
        st.balloons()
        st.markdown(f"### Summary")
        st.markdown(f"- **Total Questions:** {len(df)}")
        st.markdown(f"- **Annotated:** {len(st.session_state.annotations_submitted)}")
        if st.button("Logout and Choose Another Dataset"):
            st.session_state.teacher_name = None
            st.session_state.current_index = 0
            st.session_state.annotations_submitted = set()
            st.session_state.unannotated_rows = []
            st.rerun()
        return
    
    # Get current row from unannotated list
    if len(st.session_state.unannotated_rows) == 0:
        return
    
    current_row_number = st.session_state.unannotated_rows[0]  # Get the first unannotated row number
    
    # Find the actual dataframe index for this row number
    if 'No' in df.columns:
        row = df[df['No'] == current_row_number].iloc[0]
        current_idx = current_row_number
    else:
        row = df.iloc[current_row_number]
        current_idx = current_row_number
    
    st.title(f"Problem Set #{current_idx + 1} of {len(df)}")
    
    st.markdown("---")
    
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
        problem = row.get("Soal_x", row.get("Soal_y", row.get("Soal", "")))
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
    # Feedback Type Information Section
    with st.expander("📚 Informasi Jenis-jenis Feedback (Klik untuk melihat)", expanded=False):
        st.markdown("""
        #### Definisi Jenis-jenis Feedback:
        
        **1. Response-contingent**  
        Komentar terperinci yang meng-highlight / menyoroti respons khusus dari siswa. Bisa jadi menjelaskan mengapa jawaban yang benar adalah benar dan yang salah adalah salah. Tidak ada analisis kesalahan formal yang digunakan di sini.
        
        **2. Topic-contingent**  
        Feedback terperinci yang memberikan siswa detail tentang topik yang sedang mereka pelajari. Ini bisa berarti mengajarkan kembali materi.
        
        **3. Correct response**  
        Memberi tahu siswa jawaban yang benar untuk masalah yang diselesaikan tanpa informasi tambahan.
        
        **4. Verification**  
        Memberi tahu siswa tentang kebenaran respons mereka, seperti benar/salah atau persentase keseluruhan yang benar.
        
        **5. Try-again**  
        Memberi tahu siswa jika mereka salah menjawab dan memungkinkan siswa satu atau lebih kesempatan untuk menjawab pertanyaan.
        """)
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
        
        # Optional comments section
        st.markdown("### 💭 Additional Comments (Optional)")
        teacher_comments = st.text_area(
            "Your feedback or notes about this annotation:",
            placeholder="E.g., The feedback could be more specific about the error, or the tone seems too harsh...",
            height=100,
            help="Optional: Add any additional observations, suggestions, or notes about this feedback."
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
                "tone_style": tone_style,
                "teacher_comments": teacher_comments.strip() if teacher_comments else None
            }
            
            # Save to database
            if save_annotation(st.session_state.teacher_name, dataset_name, row, annotations):
                row_number = int(row.get('No', current_idx))
                st.session_state.annotations_submitted.add(row_number)
                # Remove this row from unannotated list
                if row_number in st.session_state.unannotated_rows:
                    st.session_state.unannotated_rows.remove(row_number)
                st.success("✅ Annotation saved successfully!")
                st.rerun()

# Main app logic
if st.session_state.teacher_name is None:
    teacher_login()
else:
    annotation_interface()
