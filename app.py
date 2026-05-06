import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from groq import Groq
from sqlalchemy import create_engine, text

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="AI SQL Data Analyst",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI SQL Data Analyst Agent")
st.markdown("Upload a CSV file and ask questions in plain English!")

# ─── Sidebar ───────────────────────────────────────────
# Works both locally and on Streamlit Cloud
    api_key = st.secrets.get("GROQ_API_KEY", "") or st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )
    model_choice = st.selectbox(
        "Choose Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192"]
    )
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Enter Groq API key")
    st.markdown("2. Upload CSV file")
    st.markdown("3. Ask any question!")

# ─── Load CSV into SQLite ──────────────────────────────
def load_csv_to_sqlite(df, db_path="data.db", table_name="data_table"):
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    return db_path, table_name

# ─── Get Table Schema ──────────────────────────────────
def get_schema(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()
    schema = f"Table: {table_name}\nColumns: "
    schema += ", ".join([f"{col[1]} ({col[2]})" for col in columns])
    return schema

# ─── Generate SQL using Groq ───────────────────────────
def generate_sql(api_key, model, schema, question):
    client = Groq(api_key=api_key)
    prompt = f"""You are a SQL expert. Given this database schema:
{schema}

Generate ONLY a valid SQLite SQL query to answer this question:
{question}

Rules:
- Return ONLY the SQL query, nothing else
- No explanations, no markdown, no backticks
- Query must end with semicolon
- Use only columns that exist in the schema"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# ─── Execute SQL ───────────────────────────────────────
def execute_sql(db_path, sql_query):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df, None
    except Exception as e:
        conn.close()
        return None, str(e)

# ─── Generate Answer using Groq ────────────────────────
def generate_answer(api_key, model, question, sql_query, result_df):
    client = Groq(api_key=api_key)
    result_str = result_df.to_string(index=False) if result_df is not None else "No results"
    prompt = f"""Question: {question}
SQL Query used: {sql_query}
Query Results:
{result_str}

Give a clear, concise answer to the question based on the results above.
Be specific with numbers and facts."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# ─── Generate Chart ────────────────────────────────────
def generate_chart(df, question):
    if df is None or len(df) == 0:
        return None

    question_lower = question.lower()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    text_cols = df.select_dtypes(include='object').columns.tolist()

    try:
        if any(w in question_lower for w in ['pie', 'share', 'percentage', 'proportion']):
            if numeric_cols and text_cols:
                return px.pie(df, values=numeric_cols[0],
                            names=text_cols[0], title="Distribution")

        elif any(w in question_lower for w in ['trend', 'over time', 'by month', 'by year']):
            if numeric_cols and text_cols:
                return px.line(df, x=text_cols[0],
                             y=numeric_cols[0], title="Trend")

        elif any(w in question_lower for w in ['top', 'highest', 'lowest', 'most', 'least', 'compare']):
            if numeric_cols and text_cols:
                return px.bar(df, x=text_cols[0],
                            y=numeric_cols[0],
                            title="Comparison",
                            color=numeric_cols[0],
                            color_continuous_scale='Blues')
        else:
            if numeric_cols and text_cols:
                return px.bar(df, x=text_cols[0],
                            y=numeric_cols[0],
                            title="Results Chart")
    except Exception:
        return None
    return None

# ─── Main App ──────────────────────────────────────────
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Data Preview
    st.subheader("📊 Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📋 Column Info"):
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Nulls': df.isnull().sum().values
        }), use_container_width=True)

    # Load to SQLite
    db_path, table_name = load_csv_to_sqlite(df)
    schema = get_schema(db_path, table_name)

    st.markdown("---")
    st.subheader("💬 Ask Your Question")

    # Sample buttons
    sample_questions = [
        "What are the top 5 products by sales?",
        "Show total sales by category",
        "Which region has the highest sales?",
        "What is average sales per month?"
    ]
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        if cols[i % 2].button(q, key=f"sq_{i}"):
            st.session_state.question = q

    question = st.text_input(
        "Your question:",
        value=st.session_state.get("question", ""),
        placeholder="e.g. What are the top 5 products by sales?"
    )

    if st.button("🔍 Analyze", type="primary") and question:
        if not api_key:
            st.error("❌ Enter your Groq API key in the sidebar!")
        else:
            with st.spinner("🤖 Generating SQL and analyzing..."):
                try:
                    # Step 1: Generate SQL
                    sql_query = generate_sql(
                        api_key, model_choice, schema, question
                    )

                    # Step 2: Execute SQL
                    result_df, error = execute_sql(db_path, sql_query)

                    # Step 3: Generate Answer
                    if error:
                        st.error(f"SQL Error: {error}")
                        st.code(sql_query, language="sql")
                    else:
                        answer = generate_answer(
                            api_key, model_choice,
                            question, sql_query, result_df
                        )

                        st.markdown("---")
                        st.subheader("✅ Results")

                        # Answer
                        st.markdown("### 💡 Answer")
                        st.success(answer)

                        # SQL
                        st.markdown("### 🗄️ SQL Query")
                        st.code(sql_query, language="sql")

                        # Chart
                        st.markdown("### 📈 Visualization")
                        chart = generate_chart(result_df, question)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.info("No chart for this query type.")

                        # Table
                        st.markdown("### 📋 Query Results")
                        st.dataframe(result_df, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Upload a CSV file to get started!")
    st.markdown("### 💡 Example Questions")
    for ex in [
        "What are the top 5 products by sales?",
        "Show total sales by region",
        "Which month had highest sales?",
        "Average sales per category?"
    ]:
        st.markdown(f"- {ex}")
