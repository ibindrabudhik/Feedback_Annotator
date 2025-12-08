# Feedback Annotation App for Teachers

A Streamlit application that allows teachers to annotate AI-generated student feedback for research purposes.

## Features

- **Teacher Authentication**: Teachers enter their name before starting
- **Dataset Selection**: Choose between Indonesian or English datasets
- **Problem Display**: View student profiles, problems, and answers
- **Chat-Style Interface**: See student-tutor conversations in a familiar chat format
- **Annotation Form**: Rate feedback on 6 criteria:
  1. Relevancy (1-4 scale)
  2. Accuracy (0-1 binary)
  3. Motivation (1-3 scale)
  4. Demotivation (1-3 scale)
  5. Guidance (1-3 scale)
  6. Tone and Style (1-4 scale)
- **Progress Tracking**: See how many problems have been annotated
- **Database Storage**: All annotations are saved to Supabase

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Supabase

1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Go to the SQL Editor and run the contents of `database_schema.sql`
4. Get your project credentials:
   - Go to Project Settings > API
   - Copy the `Project URL` and `anon/public` key

### 3. Configure Environment Variables

1. Create a `.env` file in the project directory:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Supabase credentials:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Enter Your Name**: When you first open the app, enter your name and select a dataset
2. **Review Problem**: Read the student profile, problem statement, and correct answer
3. **Analyze Conversation**: Review the student's incorrect answer and the AI tutor's feedback
4. **Rate the Feedback**: Complete all 6 rating criteria in the annotation form
5. **Submit**: Click "Submit & Next" to save and move to the next problem
6. **Track Progress**: Use the sidebar to see your annotation progress

## Database Schema

The `annotations` table stores:
- Teacher information
- Dataset and problem details
- Student answers and correct answers
- Generated feedback
- All 6 annotation ratings
- Timestamps

## Data Export

To export annotations from Supabase:

1. Go to your Supabase Dashboard
2. Navigate to Table Editor > annotations
3. Use the export feature or query via SQL:

```sql
SELECT * FROM annotations 
WHERE teacher_name = 'Your Name'
ORDER BY timestamp DESC;
```

## File Structure

```
.
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── database_schema.sql             # Supabase table schema
├── .env.example                    # Example environment variables
├── .env                           # Your actual credentials (not committed)
├── .gitignore                     # Git ignore file
├── 50_dataset_soal_indo_with_feedback_parsed_gpt_5nano.csv
└── 50_dataset_soal_indo_with_feedback_parsed.csv
```

## Rating Guidelines

### Relevancy (1-4)
- 1: Not relevant to formative feedback principles
- 2: Minimally relevant
- 3: Moderately relevant
- 4: Highly relevant and well-aligned

### Accuracy (0-1)
- 0: Incorrect or contains errors
- 1: Mathematically/factually correct

### Motivation (1-3)
- 1: Not motivating or discouraging
- 2: Neutral or somewhat motivating
- 3: Highly motivating and encouraging

### Demotivation (1-3)
- 1: Not demotivating (good)
- 2: Somewhat demotivating
- 3: Highly demotivating (poor)

### Guidance (1-3)
- 1: Provides little to no guidance
- 2: Provides some guidance
- 3: Provides clear, actionable guidance

### Tone and Style (1-4)
- 1: Poor - inappropriate or confusing
- 2: Fair - acceptable but could be better
- 3: Good - appropriate and clear
- 4: Excellent - engaging and professional

## Troubleshooting

### Database Connection Error
- Check your `.env` file has correct SUPABASE_URL and SUPABASE_KEY
- Verify your Supabase project is active
- Check if the annotations table was created successfully

### Import Errors
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.8 or higher

### CSV Not Found
- Ensure the CSV files are in the same directory as `app.py`

## Support

For issues or questions, please contact the research team.
