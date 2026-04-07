import streamlit as st
import pandas as pd
import ast
import re
import html
import math
from html.parser import HTMLParser
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

DATASET_OPTIONS = ["RAG4O", "RAG5N", "4O", "5N", "Student Raw"]
DATASET_HELP = (
    "RAG4O: RAG + GPT-4o | RAG5N: RAG + GPT-o1 | 4O: GPT-4o | "
    "5N: GPT-o1 | Student Raw: raw/student_logs + raw/tasks"
)


def normalize_row_id(value):
    """Normalize row identifiers for consistent set/list operations."""
    if value is None or pd.isna(value):
        return None

    # Keep integer-like identifiers canonical across sources: 1, 1.0, "1.0" -> "1"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass

    return text if text else None


def get_all_row_numbers(df):
    """Return row identifiers used for progress/annotation tracking."""
    if "No" in df.columns:
        ids = {normalize_row_id(v) for v in df["No"].values}
        return {v for v in ids if v is not None}
    return {str(i) for i in range(len(df))}


def get_sorted_row_numbers(df):
    """Return all row ids sorted numerically when possible."""
    ids = [normalize_row_id(v) for v in get_all_row_numbers(df)]
    ids = [v for v in ids if v is not None]

    def _sort_key(v):
        try:
            return (0, int(v))
        except (TypeError, ValueError):
            return (1, str(v))

    return sorted(ids, key=_sort_key)


def get_row_by_number(df, row_number):
    """Return dataframe row by normalized row number."""
    target = normalize_row_id(row_number)
    if target is None:
        return None

    if "No" in df.columns:
        normalized_no = df["No"].map(normalize_row_id)
        matched = df[normalized_no == target]
        if matched.empty:
            return None
        return matched.iloc[0]
    try:
        return df.iloc[int(target)]
    except (TypeError, ValueError, IndexError):
        return None


def get_navigation_order(all_rows_sorted, annotated_rows):
    """Return navigation order that prioritizes already-annotated rows first."""
    annotated_set = set(annotated_rows)
    annotated_first = [rid for rid in all_rows_sorted if rid in annotated_set]
    unannotated_after = [rid for rid in all_rows_sorted if rid not in annotated_set]
    return annotated_first + unannotated_after


def build_feedback_set_label(df, row_id, is_annotated):
    """Build a readable label for row picker in review/edit mode."""
    row = get_row_by_number(df, row_id)
    feedback_type = "N/A"
    if row is not None:
        feedback_type = str(row.get("Final_Feedback_Type", "N/A"))
    status = "Annotated" if is_annotated else "Pending"
    return f"#{row_id} | {feedback_type} | {status}"

@st.cache_data
def load_data():
    def ensure_student_raw_columns(df):
        """Ensure Student Raw editable CSV has required columns in the expected order."""
        required_cols = [
            "No",
            "Soal",
            "Jawaban",
            "Jawaban_Salah",
            "SPK",
            "SAL",
            "Final_Feedback_Type",
            "Generated_Feedback",
            "dict_generated_feedback",
        ]

        normalized = df.copy()
        for col in required_cols:
            if col not in normalized.columns:
                normalized[col] = pd.NA

        return normalized[required_cols]

    def build_student_raw_dataset(raw_dir="raw"):
        """Build an annotation-ready dataset from raw student interaction logs."""
        editable_path = os.path.join(raw_dir, "student_raw_editable.csv")

        # If editable CSV already exists, use it as the source of truth.
        if os.path.exists(editable_path):
            editable_df = pd.read_csv(editable_path)
            editable_df = ensure_student_raw_columns(editable_df)
            editable_df["SPK"] = editable_df["SPK"].fillna("N/A")
            editable_df["SAL"] = editable_df["SAL"].fillna("N/A")
            return editable_df

        logs_path = os.path.join(raw_dir, "student_logs_rows (2).csv")
        tasks_path = os.path.join(raw_dir, "tasks_rows.csv")

        logs_df = pd.read_csv(logs_path)
        tasks_df = pd.read_csv(tasks_path)

        merged_df = logs_df.merge(
            tasks_df[["task_id", "solution"]],
            on="task_id",
            how="left"
        )

        raw_df = pd.DataFrame({
            "No": range(1, len(merged_df) + 1),
            "Soal": merged_df["question"],
            "Jawaban": merged_df["solution"],
            "Jawaban_Salah": merged_df["student_answer"],
            "SPK": merged_df["achievement_level_assessed"].replace("", pd.NA).fillna(merged_df["task_level"]),
            "SAL": merged_df["error_count"],
            "Final_Feedback_Type": merged_df["feedback_type"],
            "Generated_Feedback": merged_df["feedback_given"],
            "dict_generated_feedback": pd.NA,
        })

        raw_df = ensure_student_raw_columns(raw_df)
        raw_df["SPK"] = raw_df["SPK"].fillna("N/A")
        raw_df["SAL"] = raw_df["SAL"].fillna("N/A")

        # Export first-time editable CSV so users can modify Student Raw directly.
        raw_df.to_csv(editable_path, index=False)
        return raw_df

    def sample_per_feedback_type(df, n_per_type=10):
        """Return a balanced subset with up to n_per_type rows for each feedback type."""
        feedback_col_candidates = [
            "Final_Feedback_Type",
            "feedback_type",
            "Feedback_Type",
            "jenis_feedback",
        ]

        feedback_col = next((c for c in feedback_col_candidates if c in df.columns), None)
        if feedback_col is None:
            return df

        # Keep order deterministic for reproducible annotation sessions.
        sampled = (
            df.groupby(feedback_col, group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), n_per_type), random_state=42))
            .reset_index(drop=True)
        )
        return sampled

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
            df = pd.read_csv(path)
            loaded_datasets[name] = sample_per_feedback_type(df, n_per_type=10)
        except FileNotFoundError:
            # Create dummy dataframe for testing if file doesn't exist
            df = pd.DataFrame({
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
            loaded_datasets[name] = sample_per_feedback_type(df, n_per_type=10)

    try:
        student_raw_df = build_student_raw_dataset(raw_dir="raw")
        loaded_datasets["Student Raw"] = sample_per_feedback_type(student_raw_df, n_per_type=10)
    except FileNotFoundError:
        student_raw_df = pd.DataFrame({
            'No': [1, 2, 3],
            'Soal': [f'Sample raw problem {i}' for i in range(1, 4)],
            'Jawaban': ['17', '25', '30'],
            'Jawaban_Salah': ['15', '20', '28'],
            'SPK': ['High', 'Medium', 'Low'],
            'SAL': ['1', '2', '3'], 
            'Final_Feedback_Type': ['Correct Response', 'Response Contingent', 'Topic Contingent'],
            'Generated_Feedback': [f'Sample raw feedback {i}' for i in range(1, 4)],
            'dict_generated_feedback': [pd.NA for _ in range(3)]
        })
        loaded_datasets["Student Raw"] = sample_per_feedback_type(student_raw_df, n_per_type=10)
    except Exception as e:
        st.warning(f"Could not load Student Raw dataset: {str(e)}")
    
    return loaded_datasets

datasets = load_data()

def get_annotated_rows(teacher_name, dataset_name):
    """Fetch list of row indices already annotated by this teacher for this dataset"""
    try:
        response = supabase.table("annotations").select("row_index").eq("teacher_name", teacher_name).eq("dataset_name", dataset_name).execute()
        annotated_indices = {normalize_row_id(row.get('row_index')) for row in response.data}
        return {v for v in annotated_indices if v is not None}
    except Exception as e:
        st.warning(f"Could not fetch previous annotations: {str(e)}")
        return set()


def get_existing_annotation(teacher_name, dataset_name, row_index):
    """Fetch one saved annotation for pre-filling the form."""
    try:
        response = (
            supabase.table("annotations")
            .select("relevancy,accuracy,motivation,demotivation,guidance,tone_style,teacher_comments")
            .eq("teacher_name", teacher_name)
            .eq("dataset_name", dataset_name)
            .eq("row_index", int(row_index))
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]
        return None
    except Exception:
        return None

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
if 'navigation_history' not in st.session_state:
    st.session_state.navigation_history = []
if 'form_reset_counter' not in st.session_state:
    st.session_state.form_reset_counter = 0
if 'annotation_prefill' not in st.session_state:
    st.session_state.annotation_prefill = {}
if 'all_row_numbers' not in st.session_state:
    st.session_state.all_row_numbers = []
if 'row_navigation_order' not in st.session_state:
    st.session_state.row_navigation_order = []

def chat_bubble(message, sender="user"):
    """
    sender = 'user'  -> student (right bubble, green)
    sender = 'ai'    -> tutor (left bubble, blue)
    """
    if sender == "ai":
        # Use a styled container for tutor messages with LaTeX support
        st.markdown(
            """
            <style>
            .tutor-bubble {
                background-color: #e6f2ff;
                color: black;
                padding: 15px;
                border-radius: 15px;
                max-width: 75%;
                margin: 10px 0;
                display: inline-block;
            }
            .tutor-bubble p {
                margin: 0;
                color: black;
            }
            </style>
            <div class="tutor-bubble">
                <strong>🤖 Tutor:</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Render the message with LaTeX support in a styled container
        st.markdown(f'{message}')  # LaTeX will be parsed here
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


def normalize_tutor_message(text, aggressive=True):
    """Normalize malformed tutor text so markdown/math renders more reliably."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    cleaned = str(text)
    cleaned = html.unescape(cleaned)

    # Parse noisy KaTeX/HTML payloads and keep only text + TeX annotations.
    if "<" in cleaned and ">" in cleaned:
        class _MathHTMLCleaner(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
                self.skip_katex_html_depth = 0
                self.in_tex_annotation = False
                self.math_depth = 0

            def handle_starttag(self, tag, attrs):
                if self.skip_katex_html_depth > 0:
                    self.skip_katex_html_depth += 1
                    return
                attrs_dict = dict(attrs)
                cls = attrs_dict.get("class", "")
                if "katex-html" in cls:
                    self.skip_katex_html_depth = 1
                    return
                if tag in {"script", "style"}:
                    return
                if tag == "math":
                    self.math_depth += 1
                    return
                if tag == "annotation" and attrs_dict.get("encoding", "") == "application/x-tex":
                    self.in_tex_annotation = True
                    return
                if tag in {"br", "p", "div", "li"}:
                    self.parts.append("\n")

            def handle_endtag(self, tag):
                if self.skip_katex_html_depth > 0:
                    self.skip_katex_html_depth -= 1
                    return
                if self.in_tex_annotation and tag == "annotation":
                    self.in_tex_annotation = False
                    self.parts.append(" ")
                    return
                if tag == "math" and self.math_depth > 0:
                    self.math_depth -= 1
                    return
                if tag in {"p", "div", "li"}:
                    self.parts.append("\n")

            def handle_data(self, data):
                if self.skip_katex_html_depth > 0:
                    return
                if self.math_depth > 0 and not self.in_tex_annotation:
                    return
                if data.strip():
                    self.parts.append(data)

        parser = _MathHTMLCleaner()
        parser.feed(cleaned)
        cleaned = " ".join(parser.parts)

    # Strip invisible unicode separators that often break math rendering.
    cleaned = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", cleaned)

    # Recover common escaped-sequence corruption from raw text exports.
    cleaned = cleaned.replace("\x0c", "f")  # form-feed often comes from broken \f in \frac
    cleaned = re.sub(r"\t(?=imes\b)", "t", cleaned)  # tab often comes from broken \t in \times

    # Extract TeX from MathML annotation blocks and remove any remaining tags.
    cleaned = re.sub(
        r'(?is)<annotation[^>]*encoding=["\']application/x-tex["\'][^>]*>(.*?)</annotation>',
        lambda m: f" {m.group(1)} ",
        cleaned,
    )
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)

    # Reduce over-escaped latex commands: \\frac -> \frac
    cleaned = re.sub(r"\\\\(?=[A-Za-z])", r"\\", cleaned)
    cleaned = re.sub(r"\\{2,}(?=(?:frac|times|div|ldots)\b)", r"\\", cleaned)
    # Fix whitespace after backslash: "\ frac" -> "\frac".
    cleaned = re.sub(r"\\\s+(?=(?:frac|times|div|ldots)\b)", r"\\", cleaned, flags=re.IGNORECASE)

    # Normalize escaped newlines and excessive blank spacing from CSV exports.
    cleaned = cleaned.replace("\\r\\n", "\n").replace("\\n", "\n")
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace(r"\$", "$")

    # Repair truncated latex command fragments that lost leading letters.
    cleaned = re.sub(r"(?<![A-Za-z])imes\b", "times", cleaned)
    cleaned = re.sub(r"(?<![A-Za-z])rac(?=\s*\{|\s*\d)", "frac", cleaned)

    # Repair broken latex operators with spaces/newlines between letters.
    cleaned = re.sub(r"\\\s*t\s*i\s*m\s*e\s*s", r"\\times", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\\s*d\s*i\s*v", r"\\div", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\\s*f\s*r\s*a\s*c", r"\\frac", cleaned, flags=re.IGNORECASE)

    # Remove markdown heading markers first.
    cleaned = cleaned.replace("**", "")

    if aggressive:
        # Fix "vertical" broken commands (letters split line-by-line).
        def _join_vertical_letters(word):
            pattern = r"(?is)" + r"\s*\n\s*".join(list(word))
            return re.sub(pattern, word, cleaned)

        for token in ["frac", "times", "div", "ldots"]:
            cleaned = _join_vertical_letters(token)

        # Join multi-line isolated numbers: "7\n9" -> "79".
        cleaned = re.sub(
            r"(?m)(?<!\d)(\d(?:\s*\n\s*\d)+)(?!\d)",
            lambda m: re.sub(r"\s*\n\s*", "", m.group(1)),
            cleaned,
        )

        # Normalize spaced math commands after vertical joins.
        cleaned = re.sub(r"(?i)f\s*r\s*a\s*c", "frac", cleaned)
        cleaned = re.sub(r"(?i)t\s*i\s*m\s*e\s*s", "times", cleaned)
        cleaned = re.sub(r"(?i)d\s*i\s*v", "div", cleaned)
        cleaned = re.sub(r"(?i)l\s*d\s*o\s*t\s*s", "ldots", cleaned)

        # Merge fragmented one-character lines: f\nr\na\nc -> frac, 7\n9 -> 79.
        raw_lines = [ln.strip() for ln in cleaned.split("\n")]
        merged_lines = []
        i = 0
        while i < len(raw_lines):
            current = raw_lines[i]
            if len(current) == 1 and current.isalnum():
                chars = []
                while i < len(raw_lines):
                    candidate = raw_lines[i]
                    if len(candidate) == 1 and candidate.isalnum():
                        chars.append(candidate)
                        i += 1
                    else:
                        break
                merged_lines.append("".join(chars))
                continue
            merged_lines.append(current)
            i += 1

        # Drop immediate duplicate lines often produced by corrupted exports.
        lines = merged_lines
        deduped_lines = []
        prev = None
        for line in lines:
            if line and prev and line.lower() == prev.lower():
                continue
            deduped_lines.append(line)
            prev = line if line else prev
        cleaned = "\n".join(deduped_lines)

        # Flatten single newlines to avoid vertical math fragments while keeping paragraphs.
        cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)

    # Convert shorthand fraction tokens like frac79 -> \frac{7}{9}.
    cleaned = re.sub(r"(?<!\\)frac\s*([0-9])\s*([0-9])", r"\\frac{\1}{\2}", cleaned)
    cleaned = re.sub(r"([0-9])\s*frac\s*([0-9])\s*([0-9])", r"\1 \\frac{\2}{\3}", cleaned)
    cleaned = re.sub(r"(\d+)\s*frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}", r"\1\\frac{\2}{\3}", cleaned)

    if aggressive:
        # Handle missing braces pattern: frac1times2+12 -> frac{1 times 2 + 1}{2}.
        cleaned = re.sub(
            r"(?<!\\)frac\s*([^\s]+?)\s*\+\s*(\d{1,2})",
            lambda m: f"frac{{{m.group(1)} + {m.group(2)[0]}}}{{{m.group(2)[1]}}}" if len(m.group(2)) == 2 else m.group(0),
            cleaned,
        )

    # Ensure common operators are valid LaTeX commands.
    cleaned = cleaned.replace("×", r"\times")
    cleaned = cleaned.replace("÷", r"\div")
    cleaned = re.sub(r"(?<!\\)\btimes\b", r"\\times", cleaned)
    cleaned = re.sub(r"(?<!\\)\bdiv\b", r"\\div", cleaned)
    cleaned = re.sub(r"(?<!\\)\bldots\b", r"\\ldots", cleaned)

    # Final latex normalization: repair malformed delimiters and slash fractions.
    cleaned = cleaned.replace(r"\/", r"\div")
    cleaned = cleaned.replace(r"\(", "$").replace(r"\)", "$")

    # Join split display-math expressions: $$A$$ \div $$B$$ -> $$A \div B$$.
    cleaned = re.sub(
        r"\$\$\s*(.*?)\s*\$\$\s*\\div\s*\$\$\s*(.*?)\s*\$\$",
        r"$$\1 \\div \2$$",
        cleaned,
        flags=re.DOTALL,
    )

    # Convert plain slash fractions to latex fractions.
    cleaned = re.sub(r"\(\s*([0-9]+)\s*/\s*([0-9]+)\s*\)", r"\\frac{\1}{\2}", cleaned)
    cleaned = re.sub(r"(?<![\\\d])([0-9]+)\s*/\s*([0-9]+)", r"\\frac{\1}{\2}", cleaned)

    # Fix mixed number pattern like 1(1/2) or 1\frac{1}{2}.
    cleaned = re.sub(r"(\d+)\s*\(?\s*\\frac\{(\d+)\}\{(\d+)\}\s*\)?", r"\1\\frac{\2}{\3}", cleaned)
    cleaned = re.sub(r"(\d+)\s*\(?\s*([0-9]+)\s*/\s*([0-9]+)\s*\)?", r"\1\\frac{\2}{\3}", cleaned)

    # Ensure remaining bare 'frac' tokens become proper latex command.
    cleaned = re.sub(r"(?<!\\)\bfrac\b", r"\\frac", cleaned)

    # Collapse too many spaces/newlines while preserving paragraph breaks.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_math_for_display(text, mode="latex"):
    """Format math for display.

    mode='latex': keep/promote latex rendering when possible.
    mode='plain': keep sentence readability and convert latex tokens to plain symbols.
    """
    if not text:
        return ""

    formatted = str(text)
    # Normalize common latex variants from CSV/HTML exports.
    formatted = re.sub(r"\\{2,}(?=(?:frac|times|div|ldots)\b)", r"\\", formatted)
    formatted = re.sub(r"\\\s+(?=(?:frac|times|div|ldots)\b)", r"\\", formatted, flags=re.IGNORECASE)
    formatted = formatted.replace(r"\$", "$")
    formatted = formatted.replace("×", r"\times").replace("÷", r"\div")
    formatted = formatted.replace(r"\[", "$$").replace(r"\]", "$$")
    formatted = formatted.replace(r"\(", "$").replace(r"\)", "$")

    if mode == "plain":
        # Keep prose readable, but render fractions as inline math.
        formatted = re.sub(r"\\\s*times", "×", formatted)
        formatted = re.sub(r"\\\s*div", "÷", formatted)
        formatted = re.sub(r"\\+times", "×", formatted)
        formatted = re.sub(r"\\+div", "÷", formatted)
        formatted = re.sub(r"\\+ldots", "...", formatted)

        # Mixed number latex, e.g. 1\frac{1}{7} -> $1\frac{1}{7}$
        formatted = re.sub(
            r"(\d+)\s*\\frac\{\s*(\d+)\s*\}\{\s*(\d+)\s*\}",
            r"$\1\\frac{\2}{\3}$",
            formatted,
        )

        # Standalone latex fraction, e.g. \frac{4}{9} -> $\frac{4}{9}$
        formatted = re.sub(
            r"(?<!\$)\\frac\{\s*([^{}]+?)\s*\}\{\s*([^{}]+?)\s*\}(?!\$)",
            r"$\\frac{\1}{\2}$",
            formatted,
        )

        # Fallback text fractions: (4/9) or 4/9 -> $\frac{4}{9}$
        formatted = re.sub(r"\(\s*(\d+)\s*/\s*(\d+)\s*\)", r"$\\frac{\1}{\2}$", formatted)
        formatted = re.sub(r"(?<![\d$])(\d+)\s*/\s*(\d+)(?![\d$])", r"$\\frac{\1}{\2}$", formatted)

        # Fallback mixed text fraction: 1 1/7 -> $1\frac{1}{7}$
        formatted = re.sub(r"(\d+)\s+(\d+)\s*/\s*(\d+)", r"$\1\\frac{\2}{\3}$", formatted)

        formatted = re.sub(r"[ \t]{2,}", " ", formatted)
        return formatted.strip()

    # If text contains bare latex commands but no math delimiters, promote expression to display math.
    if "$" not in formatted and re.search(r"\\(frac|times|div|ldots)\b", formatted):
        # If prose and latex commands are mixed (common after HTML <p> extraction),
        # wrap math fragments inline instead of forcing the whole sentence into $$...$$.
        if re.search(r"[A-Za-z]", formatted):
            # Normalize spacing around operators so wrapping is reliable.
            formatted = re.sub(r"\s*\\times\s*", r" \\times ", formatted, flags=re.IGNORECASE)
            formatted = re.sub(r"\s*\\div\s*", r" \\div ", formatted, flags=re.IGNORECASE)
            formatted = re.sub(r"\s*\\ldots\s*", r" \\ldots ", formatted, flags=re.IGNORECASE)

            # Wrap mixed numbers first: 4\frac{1}{2} -> $4\frac{1}{2}$
            formatted = re.sub(
                r"(?<!\$)(\d+)\s*\\frac\{\s*([^{}]+?)\s*\}\{\s*([^{}]+?)\s*\}(?!\$)",
                r"$\1\\frac{\2}{\3}$",
                formatted,
            )

            # Wrap fractions in inline math first
            formatted = re.sub(
                r"(?<!\$)\\frac\{\s*([^{}]+?)\s*\}\{\s*([^{}]+?)\s*\}(?!\$)",
                r"$\\frac{\1}{\2}$",
                formatted,
            )
            
            # Wrap operators: match with flexible spacing (before/after or in groups)
            # Pattern 1: \times with spaces before and/or after
            formatted = re.sub(r"(?<!\$)\s*\\times\s*(?!\$)", r" $\\times$ ", formatted, flags=re.IGNORECASE)
            formatted = re.sub(r"(?<!\$)\s*\\div\s*(?!\$)", r" $\\div$ ", formatted, flags=re.IGNORECASE)
            formatted = re.sub(r"(?<!\$)\s*\\ldots\s*(?!\$)", r" $\\ldots$ ", formatted, flags=re.IGNORECASE)
            
            # Simplify: ensure every \times, \div, \ldots not in $ is wrapped
            formatted = re.sub(r"(?<!\$)\\times(?!\$)", r"$\\times$", formatted, flags=re.IGNORECASE)
            formatted = re.sub(r"(?<!\$)\\div(?!\$)", r"$\\div$", formatted, flags=re.IGNORECASE)
            formatted = re.sub(r"(?<!\$)\\ldots(?!\$)", r"$\\ldots$", formatted, flags=re.IGNORECASE)

            # Clean up spacing: collapse multiple spaces and fix $ edges
            formatted = re.sub(r"\$\s+", "$", formatted)
            formatted = re.sub(r"\s+\$", "$", formatted)
            formatted = re.sub(r"\$\$+", "$$", formatted)
            formatted = re.sub(r"[ ]{2,}", " ", formatted)  # Collapse multiple spaces
            return formatted

        match = re.match(r"^\s*([^\n:]+:)\s*(.+)$", formatted, flags=re.DOTALL)
        if match:
            prefix = match.group(1).strip()
            expr = match.group(2).strip()
            return f"{prefix}\n\n$${expr}$$"
        return f"$${formatted.strip()}$$"

    # If proper math delimiters exist, let Streamlit render them directly.
    if "$" in formatted:
        return formatted

    # Fallback for bare commands that would otherwise display literally.
    formatted = re.sub(r"\\+times", "×", formatted)
    formatted = re.sub(r"\\+div", "÷", formatted)
    formatted = re.sub(r"\\+ldots", "...", formatted)

    # Convert fractions into readable a/b text form.
    formatted = re.sub(r"\\frac\{\s*([^{}]+?)\s*\}\{\s*([^{}]+?)\s*\}", r"(\1/\2)", formatted)
    formatted = re.sub(r"(\d+)\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)", r"\1 \2/\3", formatted)
    return formatted

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
        
        # Use upsert so teachers can re-annotate previously submitted rows.
        response = (
            supabase.table("annotations")
            .upsert(data, on_conflict="teacher_name,dataset_name,row_index")
            .execute()
        )
        return True
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error saving to database: {error_msg}")
        return False

def teacher_login():
    """Teacher name entry screen"""
    st.title("📝 Feedback Annotation System")
    st.markdown("### Welcome, Teacher!")
    st.markdown("Silahkan masukkan nama Anda dan pilih dataset yang ingin Anda anotasi. Setelah itu, Anda akan diarahkan ke halaman anotasi di mana Anda dapat memberikan penilaian pada feedback yang diberikan kepada siswa.")
    #Button to stop the app
    if st.button("Stop Anotasi"):
        st.stop()
    
    with st.form("teacher_login_form"):
        teacher_name = st.text_input("Nama:", placeholder="e.g., Dr. Smith")
        dataset_choice = st.radio("Pilih Dataset:", 
                                  DATASET_OPTIONS,
                                  help=DATASET_HELP)
        submit = st.form_submit_button("Mulai Anotasi")
        
        if submit and teacher_name:
            st.session_state.teacher_name = teacher_name.strip()
            st.session_state.selected_dataset = dataset_choice
            st.session_state.current_index = 0
            st.session_state.navigation_history = []
            st.session_state.form_reset_counter = 0
            st.session_state.annotation_prefill = {}
            
            # Fetch already-annotated questions for this teacher and dataset
            annotated = get_annotated_rows(teacher_name.strip(), dataset_choice)
            # Keep only annotations that exist in the currently loaded dataset.
            df = datasets[dataset_choice]
            all_row_numbers = get_all_row_numbers(df)
            annotated_in_dataset = {rid for rid in annotated if rid in all_row_numbers}
            st.session_state.annotations_submitted = annotated_in_dataset
            
            # Get list of unannotated rows
            st.session_state.all_row_numbers = get_sorted_row_numbers(df)
            st.session_state.row_navigation_order = get_navigation_order(
                st.session_state.all_row_numbers,
                annotated_in_dataset,
            )
            
            unannotated = [rid for rid in st.session_state.all_row_numbers if rid not in annotated_in_dataset]
            st.session_state.unannotated_rows = unannotated
            
            st.session_state.current_index = (
                st.session_state.row_navigation_order[0]
                if st.session_state.row_navigation_order
                else 0
            )
            
            st.rerun()
        elif submit:
            st.error("Silahkan masukkan nama anda.")

def annotation_interface():
    """Main annotation interface"""
    # Get current dataset
    dataset_name = st.session_state.selected_dataset
    df = datasets[dataset_name]
    
    # Get unannotated rows
    if not st.session_state.unannotated_rows:
        all_rows = get_all_row_numbers(df)
        all_sorted = get_sorted_row_numbers(df)
        unannotated = [rid for rid in all_sorted if rid not in st.session_state.annotations_submitted]
        st.session_state.unannotated_rows = unannotated

    st.session_state.all_row_numbers = get_sorted_row_numbers(df)
    if not st.session_state.all_row_numbers:
        st.warning("Dataset tidak memiliki baris yang bisa dianotasi.")
        return

    st.session_state.row_navigation_order = get_navigation_order(
        st.session_state.all_row_numbers,
        st.session_state.annotations_submitted,
    )
    if not st.session_state.row_navigation_order:
        st.warning("Navigasi data tidak tersedia.")
        return

    normalized_current = normalize_row_id(st.session_state.current_index)
    if normalized_current not in st.session_state.row_navigation_order:
        st.session_state.current_index = st.session_state.row_navigation_order[0]
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👨‍🏫 Teacher: {st.session_state.teacher_name}")
        
        st.markdown("---")
        st.markdown("### 📊 Ganti Dataset")
        
        # Dataset selector
        new_dataset = st.selectbox(
            "Pilih Dataset:",
            DATASET_OPTIONS,
            index=DATASET_OPTIONS.index(st.session_state.selected_dataset),
            help=DATASET_HELP
        )
        
        # Check if dataset changed
        if new_dataset != st.session_state.selected_dataset:
            st.session_state.selected_dataset = new_dataset
            st.session_state.navigation_history = []
            st.session_state.form_reset_counter = 0
            st.session_state.annotation_prefill = {}
            
            # Fetch annotations for new dataset
            annotated = get_annotated_rows(st.session_state.teacher_name, new_dataset)
            # Keep only annotations that exist in the currently loaded dataset.
            new_df = datasets[new_dataset]
            all_row_numbers = get_all_row_numbers(new_df)
            annotated_in_dataset = {rid for rid in annotated if rid in all_row_numbers}
            st.session_state.annotations_submitted = annotated_in_dataset
            
            # Get list of unannotated rows for new dataset
            st.session_state.all_row_numbers = get_sorted_row_numbers(new_df)
            st.session_state.row_navigation_order = get_navigation_order(
                st.session_state.all_row_numbers,
                annotated_in_dataset,
            )
            
            unannotated = [rid for rid in st.session_state.all_row_numbers if rid not in annotated_in_dataset]
            st.session_state.unannotated_rows = unannotated
            
            st.session_state.current_index = (
                st.session_state.row_navigation_order[0]
                if st.session_state.row_navigation_order
                else 0
            )
            
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"**Dataset saat ini:** {dataset_name}")
        all_row_set = set(st.session_state.all_row_numbers)
        annotated_in_dataset = st.session_state.annotations_submitted.intersection(all_row_set)
        total_annotated = len(annotated_in_dataset)
        total_questions = len(df)
        remaining = len(st.session_state.unannotated_rows)
        st.markdown(f"**Progress:** {total_annotated}/{total_questions} annotated")
        st.markdown(f"**Tersisa:** {remaining} pertanyaan dan feedback")
        progress_value = (total_annotated / total_questions) if total_questions > 0 else 0.0
        st.progress(min(max(progress_value, 0.0), 1.0))

        st.markdown("---")
        st.markdown("### 🛠️ Pilih Feedback untuk Edit")
        current_row_for_picker = normalize_row_id(st.session_state.current_index)
        if current_row_for_picker not in st.session_state.row_navigation_order:
            current_row_for_picker = st.session_state.row_navigation_order[0]

        picker_index = st.session_state.row_navigation_order.index(current_row_for_picker)
        selected_row = st.selectbox(
            "Pilih nomor problem/feedback:",
            st.session_state.row_navigation_order,
            index=picker_index,
            format_func=lambda rid: build_feedback_set_label(
                df,
                rid,
                rid in st.session_state.annotations_submitted,
            ),
            help="Anda bisa memilih data mana pun untuk dilihat dan diedit ulang, termasuk yang sudah Submit All.",
        )

        selected_normalized = normalize_row_id(selected_row)
        if selected_normalized != normalize_row_id(st.session_state.current_index):
            st.session_state.current_index = selected_normalized
            st.rerun()

        feedback_col = "Final_Feedback_Type" if "Final_Feedback_Type" in df.columns else None
        if feedback_col:
            counts_df = (
                df[feedback_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis("Feedback Type")
                .reset_index(name="Count")
            )
            st.markdown("### 🧾 Distribusi Feedback")
            if dataset_name == "Student Raw":
                st.caption("Balanced sample: maksimum 10 data per tipe feedback.")
            st.dataframe(counts_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.teacher_name = None
            st.session_state.current_index = 0
            st.session_state.annotations_submitted = set()
            st.session_state.unannotated_rows = []
            st.session_state.navigation_history = []
            st.session_state.form_reset_counter = 0
            st.session_state.annotation_prefill = {}
            st.session_state.row_navigation_order = []
            st.rerun()
    
    # Check if all done
    if len(st.session_state.unannotated_rows) == 0:
        st.success("🎉 Semua data sudah dianotasi. Anda sekarang di mode review.")
        st.markdown(f"### Summary")
        st.markdown(f"- **Total Questions:** {len(df)}")
        st.markdown(f"- **Annotated:** {len(st.session_state.annotations_submitted)}")
        if st.session_state.all_row_numbers:
            current_norm = normalize_row_id(st.session_state.current_index)
            if current_norm not in st.session_state.row_navigation_order:
                st.session_state.current_index = st.session_state.row_navigation_order[0]
        if st.button("Logout and Choose Another Dataset"):
            st.session_state.teacher_name = None
            st.session_state.current_index = 0
            st.session_state.annotations_submitted = set()
            st.session_state.unannotated_rows = []
            st.session_state.navigation_history = []
            st.session_state.form_reset_counter = 0
            st.session_state.annotation_prefill = {}
            st.session_state.row_navigation_order = []
            st.rerun()
    
    current_row_number = normalize_row_id(st.session_state.current_index)
    if current_row_number is None:
        current_row_number = st.session_state.row_navigation_order[0]
        st.session_state.current_index = current_row_number

    row = get_row_by_number(df, current_row_number)
    if row is None:
        st.error("Row tidak ditemukan pada dataset saat ini.")
        return
    current_idx = current_row_number
    
    st.title(f"Problem Set #{current_idx} of {len(df)}")
    
    st.markdown("---")
    st.info(f"Anda diminta untuk melakukan anotasi, lihat informasi berikut terkait jenis feedback yang diberikan kepada siswa. Kemudian, berikan penilaian Anda berdasarkan kriteria yang tersedia di bawah ini.")
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
    
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Student Profile & Problem Details
        st.markdown("### 📋 Student Profile & Problem Details")
        problem_number_display = row.get('No')
        if pd.isna(problem_number_display):
            problem_number_display = current_idx
        
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px; margin-bottom: 15px; color: black;">
            <strong>Problem Number:</strong> {problem_number_display}<br>
            <strong>Knowledge Level (SPK):</strong> {row.get('SPK', 'N/A')}<br>
            <strong>Mistake Level (SAL):</strong> {row.get('SAL', 'N/A')}<br>
            <strong>Feedback Type:</strong> {row.get('Final_Feedback_Type', 'N/A')}<br>
        </div>
        """, unsafe_allow_html=True)
        
        # Problem Statement (rendered with native markdown so LaTeX is parsed)
        problem = row.get("Soal_x", row.get("Soal_y", row.get("Soal", "")))
        problem = normalize_tutor_message(problem, aggressive=True)
        problem = format_math_for_display(problem, mode="latex")
        with st.container(border=True):
            st.markdown("**📝 Problem:**")
            st.markdown(problem)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Correct Answer (rendered with native markdown so LaTeX is parsed)
        correct_answer = row.get("Jawaban", row.get("correct_answer", ""))
        correct_answer = normalize_tutor_message(correct_answer, aggressive=True)
        correct_answer = format_math_for_display(correct_answer, mode="latex")
        with st.container(border=True):
            st.markdown("**✅ Correct Answer:**")
            st.markdown(correct_answer)
    
    with col2:
        # Chat-style conversation
        st.markdown("### 💬 Percakapan Student-Tutor")
        
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

        feedback_text = normalize_tutor_message(feedback_text)
        has_math_markers = bool(
            re.search(r"\$|\\\(|\\\)|\\\[|\\\]|\\(frac|times|div|ldots)\b", feedback_text)
        )
        feedback_mode = "latex" if has_math_markers else "plain"
        feedback_text = format_math_for_display(feedback_text, mode=feedback_mode)
        
        student_answer = row.get("Jawaban_Salah", row.get("student_error", ""))
        
        # Display chat
        chat_bubble(student_answer, sender="user")
        chat_bubble(feedback_text, sender="ai")

    # Annotation Form
    st.markdown("---")
    
    st.markdown("### 📊 Annotation Form")
    st.markdown("Silahkan beri penilaian pada feedback yang diberikan berdasarkan kriteria berikut. Anda juga dapat menambahkan komentar tambahan di bagian bawah jika diperlukan.")
    
    form_suffix = st.session_state.form_reset_counter
    prefill_key = f"{dataset_name}::{current_row_number}"
    prefill = st.session_state.annotation_prefill.get(prefill_key)
    if prefill is None:
        loaded_prefill = get_existing_annotation(
            st.session_state.teacher_name,
            dataset_name,
            current_row_number,
        )
        if loaded_prefill is None:
            loaded_prefill = {}
        st.session_state.annotation_prefill[prefill_key] = loaded_prefill
        prefill = loaded_prefill

    def _idx(options, value, fallback=0):
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return options.index(normalized) if normalized in options else fallback

    with st.form("annotation_form"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            relevancy = st.radio(
                "1. Relevansi dengan Feedback Formative",
                options=[1, 2, 3, 4],
                index=_idx([1, 2, 3, 4], prefill.get("relevancy"), fallback=0),
                key=f"relevancy_{dataset_name}_{current_row_number}_{form_suffix}",
                format_func=lambda x: f"{x} - {'Sangat Rendah' if x==1 else 'Rendah' if x==2 else 'Tinggi' if x==3 else 'Sangat Tinggi'}",
                help="Seberapa relevan feedback ini dengan prinsip penilaian formatif?"
            )
            
            accuracy = st.radio(
                "2. Akurasinya",
                options=[0, 1],
                index=_idx([0, 1], prefill.get("accuracy"), fallback=0),
                key=f"accuracy_{dataset_name}_{current_row_number}_{form_suffix}",
                format_func=lambda x: "Salah" if x == 0 else "Benar",
                help="Apakah feedback ini akurat secara matematis/fakta?"
            )
        
        with col_b:
            motivation = st.radio(
                "3. Motivation",
                options=[1, 2, 3],
                index=_idx([1, 2, 3], prefill.get("motivation"), fallback=0),
                key=f"motivation_{dataset_name}_{current_row_number}_{form_suffix}",
                format_func=lambda x: f"{x} - {'Rendah' if x==1 else 'Sedang' if x==2 else 'Tinggi'}",
                help="Seberapa memotivasi feedback ini bagi siswa?"
            )
            
            demotivation = st.radio(
                "4. Demotivation",
                options=[1, 2, 3],
                index=_idx([1, 2, 3], prefill.get("demotivation"), fallback=0),
                key=f"demotivation_{dataset_name}_{current_row_number}_{form_suffix}",
                format_func=lambda x: f"{x} - {'Rendah' if x==1 else 'Sedang' if x==2 else 'Tinggi'}",
                help="Seberapa demotivasi feedback ini bagi siswa? (Semakin rendah semakin baik)"
            )
        
        with col_c:
            guidance = st.radio(
                "5. Guidance",
                options=[1, 2, 3],
                index=_idx([1, 2, 3], prefill.get("guidance"), fallback=0),
                key=f"guidance_{dataset_name}_{current_row_number}_{form_suffix}",
                format_func=lambda x: f"{x} - {'Rendah' if x==1 else 'Sedang' if x==2 else 'Tinggi'}",
                help="Seberapa baik feedback ini membimbing siswa?"
            )
            
            tone_style = st.radio(
                "6. Tone and Style",
                options=[1, 2, 3, 4],
                index=_idx([1, 2, 3, 4], prefill.get("tone_style"), fallback=0),
                key=f"tone_style_{dataset_name}_{current_row_number}_{form_suffix}",
                format_func=lambda x: f"{x} - {'Buruk' if x==1 else 'Cukup' if x==2 else 'Baik' if x==3 else 'Sangat Baik'}",
                help="Seberapa baik tone dan gaya dari feedback ini?"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Optional comments section
        st.markdown("### 💭 Komentar (Opsional)")
        teacher_comments = st.text_area(
            "Komentar Anda tentang anotasi ini:",
            value=prefill.get("teacher_comments") or "",
            key=f"teacher_comments_{dataset_name}_{current_row_number}_{form_suffix}",
            placeholder="Mis. feedback bisa lebih spesifik tentang kesalahan, atau tone terlalu tajam...",
            height=100,
            help="Optional: Berikan tambahan observasi, saran, atau catatan tentang anotasi ini."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        nav_index = st.session_state.row_navigation_order.index(current_row_number)
        back_disabled = nav_index <= 0

        col_back, col_submit, col_submit_all = st.columns([1, 1, 2])
        with col_back:
            back_button = st.form_submit_button("⬅️ Back", use_container_width=True, disabled=back_disabled)
        with col_submit:
            submit_button = st.form_submit_button("✅ Submit & Next", use_container_width=True)
        with col_submit_all:
            submit_all_button = st.form_submit_button("📦 Submit All", use_container_width=True)

        if back_button:
            prev_row = st.session_state.row_navigation_order[nav_index - 1]
            st.session_state.current_index = prev_row
            st.rerun()

        if submit_button or submit_all_button:
            annotations = {
                "relevancy": relevancy,
                "accuracy": accuracy,
                "motivation": motivation,
                "demotivation": demotivation,
                "guidance": guidance,
                "tone_style": tone_style,
                "teacher_comments": teacher_comments.strip() if teacher_comments else None
            }
            
            def _save_one(target_row_number):
                target_row = get_row_by_number(df, target_row_number)
                if target_row is None:
                    return False
                ok = save_annotation(st.session_state.teacher_name, dataset_name, target_row, annotations)
                if not ok:
                    return False

                st.session_state.annotation_prefill[f"{dataset_name}::{target_row_number}"] = {
                    "relevancy": relevancy,
                    "accuracy": accuracy,
                    "motivation": motivation,
                    "demotivation": demotivation,
                    "guidance": guidance,
                    "tone_style": tone_style,
                    "teacher_comments": teacher_comments.strip() if teacher_comments else None,
                }
                st.session_state.annotations_submitted.add(target_row_number)
                if target_row_number in st.session_state.unannotated_rows:
                    st.session_state.unannotated_rows.remove(target_row_number)
                return True

            if submit_all_button:
                remaining_rows = [current_row_number] + [
                    rid for rid in st.session_state.unannotated_rows if rid != current_row_number
                ]
                saved_count = 0
                for rid in remaining_rows:
                    if _save_one(rid):
                        saved_count += 1
                    else:
                        break

                st.session_state.form_reset_counter += 1
                if saved_count == len(remaining_rows):
                    if st.session_state.row_navigation_order:
                        st.session_state.current_index = st.session_state.row_navigation_order[-1]
                    st.success(f"✅ Submit All berhasil: {saved_count} data disimpan.")
                else:
                    st.warning(f"⚠️ Submit All berhenti setelah {saved_count} data.")
                st.rerun()

            if submit_button and _save_one(current_row_number):
                st.session_state.form_reset_counter += 1

                following = st.session_state.row_navigation_order[nav_index + 1:]
                if following:
                    st.session_state.current_index = following[0]
                else:
                    st.session_state.current_index = st.session_state.row_navigation_order[0]

                st.success("✅ Anotasi berhasil disimpan!")
                st.rerun()

# Main app logic
if st.session_state.teacher_name is None:
    teacher_login()
else:
    annotation_interface()