"""
Data processing functions for TAP AI Assistant
"""
import os
import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

from config import DATA_DIR
from config import REFERENCE_DATE

def count_students_improved(program: str, days: int = 90):
    """
    Count new enrollments in a program over the past N days
    
    Args:
        program: Program name to filter by
        days: Number of days to look back
        
    Returns:
        Number of students enrolled in the program within the specified time period
    """
    try:
        # Use absolute path to ensure file is found
        file_path = os.path.join(os.path.abspath(DATA_DIR), "student_dataset.csv")
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return 0
            
        df = pd.read_csv(
            file_path,
            parse_dates=["enrollment_date"],
        )
        
        # For demo purposes, use a reference date based on the data
        reference_date = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d")  # Current date
        start = reference_date - relativedelta(days=days)
        
        # Filter by program first (case-insensitive)
        df_prog = df[df["program"].str.contains(program, case=False, na=False)]
        
        if df_prog.empty:
            st.warning(f"No students found in the {program} program")
            return 0
        
        # Use error handling when filtering by date in case of parsing issues
        try:
            # Filter only rows with valid dates
            df_prog = df_prog.dropna(subset=["enrollment_date"])
            recent = df_prog[(df_prog["enrollment_date"] >= start) & (df_prog["enrollment_date"] <= reference_date)]
            return len(recent)
        except Exception as e:
            st.error(f"Date filtering error: {e}")
            return 0
    except Exception as e:
        st.error(f"Error counting students: {e}")
        return 0

def find_teachers_missing_feedback(days: int = 14):
    """
    Find teachers missing feedback in past N days
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of teacher names who haven't submitted feedback
    """
    try:
        # Use absolute path for file
        file_path = os.path.join(os.path.abspath(DATA_DIR), "teacher_dataset.csv")
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return ["Error: teacher dataset file not found"]
            
        # Load the dataset and ensure date parsing
        try:
            df = pd.read_csv(file_path)
            # Convert date strings to datetime objects
            if "last_feedback_date" in df.columns:
                df["last_feedback_date"] = pd.to_datetime(df["last_feedback_date"], errors='coerce')
            else:
                st.error("Column 'last_feedback_date' not found in teacher dataset")
                return ["Error: last_feedback_date column not found"]
        except Exception as e:
            st.error(f"Error parsing teacher dataset: {e}")
            return [f"Error parsing teacher dataset: {e}"]
        
        # Use the reference date matching our dataset
        reference_date = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d")  # Current date
        threshold = reference_date - relativedelta(days=days)
        
        # Make sure feedback_submitted column exists and handle missing values
        if "feedback_submitted" not in df.columns:
            st.error("Column 'feedback_submitted' not found in teacher dataset")
            return ["Error: feedback_submitted column not found"]
        
        # Handle potential NaN values in feedback_submitted
        df["feedback_submitted"] = df["feedback_submitted"].fillna("No")
        
        try:
            # Filter those who haven't submitted feedback recently
            missing = df[
                (df["feedback_submitted"].str.lower() == "no") |
                (df["last_feedback_date"] < threshold)
            ]
            
            # Check if we have any results
            if missing.empty:
                return ["No teachers found missing feedback in the specified period"]
                
            # Return just the names
            return missing["name"].tolist()
        except Exception as e:
            st.error(f"Error filtering teacher feedback: {e}")
            return [f"Error processing teacher data: {e}"]
    except Exception as e:
        st.error(f"Error loading teacher data: {e}")
        return [f"Data loading error: {e}"]

def filter_students_by_name_start(start_letter):
    """
    Count students whose names start with a specific letter
    
    Args:
        start_letter: The letter to filter by
        
    Returns:
        Tuple containing (count of students, list of names)
    """
    try:
        file_path = os.path.join(os.path.abspath(DATA_DIR), "student_dataset.csv")
        if not os.path.exists(file_path):
            return (0, [])
            
        student_df = pd.read_csv(file_path)
        
        # Make sure the name column exists
        if 'name' not in student_df.columns:
            st.error("The 'name' column is missing from student dataset")
            return (0, [])
        
        # Filter students with names starting with the letter (case insensitive)
        filtered_df = student_df[student_df['name'].str.lower().str.startswith(start_letter.lower(), na=False)]
        
        # Get the total count and list of names
        count = len(filtered_df)
        names = filtered_df['name'].tolist() if count <= 10 else filtered_df['name'].tolist()[:10]
        
        return (count, names)
    except Exception as e:
        st.error(f"Error filtering students by name: {e}")
        return (0, [])

def analyze_dataset(query):
    """
    Dynamically analyze student and teacher datasets to provide comprehensive data to the LLM
    
    Args:
        query: User's query string
        
    Returns:
        String containing analysis context for the LLM
    """
    result = {}
    student_data_info = {}
    teacher_data_info = {}
    
    try:
        # Load datasets
        student_file = os.path.join(os.path.abspath(DATA_DIR), "student_dataset.csv")
        teacher_file = os.path.join(os.path.abspath(DATA_DIR), "teacher_dataset.csv")
        
        # Process student dataset if it exists
        if os.path.exists(student_file):
            student_df = pd.read_csv(student_file)
            student_data_info['total_count'] = len(student_df)
            student_data_info['columns'] = list(student_df.columns)
            
            # Provide sample data for the LLM to understand the structure
            student_data_info['sample'] = student_df.head(3).to_dict(orient='records')
            
            # Include basic aggregations that might be useful
            if 'coding_score' in student_df.columns:
                student_data_info['avg_coding_score'] = round(student_df['coding_score'].mean(), 2)
                student_data_info['min_coding_score'] = int(student_df['coding_score'].min())
                student_data_info['max_coding_score'] = int(student_df['coding_score'].max())
            
            if 'math_score' in student_df.columns:
                student_data_info['avg_math_score'] = round(student_df['math_score'].mean(), 2)
                student_data_info['min_math_score'] = int(student_df['math_score'].min())
                student_data_info['max_math_score'] = int(student_df['math_score'].max())
            
            if 'attendance_percent' in student_df.columns:
                student_data_info['avg_attendance'] = round(student_df['attendance_percent'].mean(), 2)
            
            # Get unique values for categorical fields (with counts)
            for col in ['program', 'location', 'gender', 'term']:
                if col in student_df.columns:
                    value_counts = student_df[col].value_counts().to_dict()
                    student_data_info[f'{col}_counts'] = value_counts
        
        # Process teacher dataset if it exists
        if os.path.exists(teacher_file):
            teacher_df = pd.read_csv(teacher_file)
            teacher_data_info['total_count'] = len(teacher_df)
            teacher_data_info['columns'] = list(teacher_df.columns)
            
            # Provide sample data for the LLM to understand the structure
            teacher_data_info['sample'] = teacher_df.head(3).to_dict(orient='records')
            
            # Get unique values for categorical fields (with counts)
            for col in ['subject', 'status', 'location']:
                if col in teacher_df.columns:
                    value_counts = teacher_df[col].value_counts().to_dict()
                    teacher_data_info[f'{col}_counts'] = value_counts
                    
            if 'feedback_submitted' in teacher_df.columns:
                feedback_counts = teacher_df['feedback_submitted'].value_counts().to_dict()
                teacher_data_info['feedback_submitted_counts'] = feedback_counts
        
        # Create a context string with detailed information about the datasets
        context = "# Dataset Information\n\n"
        
        # Student dataset information
        if student_data_info:
            context += "## Student Dataset\n"
            context += f"- Total students: {student_data_info.get('total_count', 'N/A')}\n"
            context += f"- Available columns: {', '.join(student_data_info.get('columns', []))}\n\n"
            
            # Add categorical distributions
            for category in ['program', 'location', 'gender', 'term']:
                counts_key = f'{category}_counts'
                if counts_key in student_data_info:
                    context += f"### {category.capitalize()} Distribution:\n"
                    for value, count in student_data_info[counts_key].items():
                        context += f"- {value}: {count} students\n"
                    context += "\n"
            
            # Add score statistics
            context += "### Performance Metrics:\n"
            if 'avg_coding_score' in student_data_info:
                context += f"- Coding scores: avg={student_data_info['avg_coding_score']}, "
                context += f"range={student_data_info['min_coding_score']}-{student_data_info['max_coding_score']}\n"
            
            if 'avg_math_score' in student_data_info:
                context += f"- Math scores: avg={student_data_info['avg_math_score']}, "
                context += f"range={student_data_info['min_math_score']}-{student_data_info['max_math_score']}\n"
            
            if 'avg_attendance' in student_data_info:
                context += f"- Average attendance: {student_data_info['avg_attendance']}%\n\n"
            
            # Add a sample of student data
            context += "### Sample Student Records:\n"
            if 'sample' in student_data_info:
                for i, student in enumerate(student_data_info['sample']):
                    context += f"Student {i+1}: {student}\n"
                context += "\n"
        
        # Teacher dataset information
        if teacher_data_info:
            context += "## Teacher Dataset\n"
            context += f"- Total teachers: {teacher_data_info.get('total_count', 'N/A')}\n"
            context += f"- Available columns: {', '.join(teacher_data_info.get('columns', []))}\n\n"
            
            # Add categorical distributions
            for category in ['subject', 'status', 'location']:
                counts_key = f'{category}_counts'
                if counts_key in teacher_data_info:
                    context += f"### {category.capitalize()} Distribution:\n"
                    for value, count in teacher_data_info[counts_key].items():
                        context += f"- {value}: {count} teachers\n"
                    context += "\n"
            
            if 'feedback_submitted_counts' in teacher_data_info:
                context += "### Feedback Submission Status:\n"
                for status, count in teacher_data_info['feedback_submitted_counts'].items():
                    context += f"- {status}: {count} teachers\n"
                context += "\n"
            
            # Add a sample of teacher data
            context += "### Sample Teacher Records:\n"
            if 'sample' in teacher_data_info:
                for i, teacher in enumerate(teacher_data_info['sample']):
                    context += f"Teacher {i+1}: {teacher}\n"
                context += "\n"
        
        # Add a note on how to interpret the data
        context += "\n## Query Information\n"
        context += f"- User query: \"{query}\"\n"
        context += f"- The current date is {REFERENCE_DATE}.\n\n"
        
        # Add SQL-like capabilities explanation
        context += "## Data Analysis Capabilities\n"
        context += "You can answer the question by analyzing the datasets as if you could perform SQL-like operations. "
        context += "You can filter, count, aggregate, and analyze the data based on various conditions mentioned in the query. "
        context += "For example, you can count students whose names start with a specific letter, find teachers in particular locations, "
        context += "calculate average scores for specific student groups, etc.\n\n"
        
        return context
    except Exception as e:
        import traceback
        print(f"Error in analyze_dataset: {str(e)}")
        print(traceback.format_exc())
        return f"Error analyzing datasets: {str(e)}"