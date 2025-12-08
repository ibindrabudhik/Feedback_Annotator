-- SQL Schema for Supabase Database
-- Run this in your Supabase SQL Editor to create the annotations table

CREATE TABLE IF NOT EXISTS annotations (
    id BIGSERIAL PRIMARY KEY,
    teacher_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    problem TEXT,
    student_answer TEXT,
    correct_answer TEXT,
    generated_feedback TEXT,
    relevancy INTEGER CHECK (relevancy >= 1 AND relevancy <= 4),
    accuracy INTEGER CHECK (accuracy IN (0, 1)),
    motivation INTEGER CHECK (motivation >= 1 AND motivation <= 3),
    demotivation INTEGER CHECK (demotivation >= 1 AND demotivation <= 3),
    guidance INTEGER CHECK (guidance >= 1 AND guidance <= 3),
    tone_style INTEGER CHECK (tone_style >= 1 AND tone_style <= 4),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_teacher_name ON annotations(teacher_name);
CREATE INDEX IF NOT EXISTS idx_dataset_name ON annotations(dataset_name);
CREATE INDEX IF NOT EXISTS idx_timestamp ON annotations(timestamp);

-- Enable Row Level Security (RLS)
ALTER TABLE annotations ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (adjust based on your security needs)
CREATE POLICY "Enable all operations for authenticated users" ON annotations
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Or if you want public access (not recommended for production):
CREATE POLICY "Enable insert for all users" ON annotations
    FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Enable read for all users" ON annotations
    FOR SELECT
    USING (true);
