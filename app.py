import os
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="ChatExcelCSV", page_icon="📝", layout="wide", initial_sidebar_state="auto", menu_items=None)

# Create memory for the conversation
memory = ConversationBufferMemory()

# Define file_formats outside of any function
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    """Clear the Submit Button State"""
    st.session_state["submit"] = False

@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        try:
            df = file_formats[ext](uploaded_file)
            # Fill NA values with appropriate replacements
            df = df.fillna({
                col: 0 if pd.api.types.is_numeric_dtype(df[col]) else 'Unknown'
                for col in df.columns
            })
            return df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

def generate_context_prompt(dataframes):
    context = "You are working with multiple dataframes. Here's a summary of the data:\n\n"
    for name, df in dataframes.items():
        context += f"Dataframe '{name}':\n"
        context += f"- Shape: {df.shape}\n"
        context += f"- Columns: {', '.join(df.columns)}\n\n"
    context += "Please refer to the dataframes by their names when answering questions.\n"
    context += "Now, please answer the following question about the data:\n\n"
    return context

def combine_dataframes(dataframes):
    """
    Combine multiple dataframes into a single dataframe with a 'source' column
    indicating which file it came from. Handles null values appropriately.
    """
    if not dataframes:
        return pd.DataFrame()
    
    combined_dfs = []
    for name, df in dataframes.items():
        df_copy = df.copy()
        # Fill NA values appropriately
        for col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(0)
            else:
                df_copy[col] = df_copy[col].fillna('Unknown')
        
        df_copy['source'] = name
        combined_dfs.append(df_copy)
    
    # Combine all dataframes
    try:
        result = pd.concat(combined_dfs, axis=0, ignore_index=True)
        return result
    except Exception as e:
        st.error(f"Error combining dataframes: {str(e)}")
        return pd.DataFrame()

def main():
    st.header("Combine & Summarise: Chat with your Excel or CSV files")
    st.subheader("Merge Multiple Datasets for Comprehensive Reports")

    with st.sidebar:
        st.header("Settings")
        
        # Add API key input to sidebar
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        
        # Add model selection dropdown
        model_name = st.selectbox(
            "Select GPT Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0
        )
        
        st.header("Description")
        st.write(
            '''<p style="font-size: small;">
        <div class="user-story">
        <strong>User Story:</strong><br>
        As a business analyst, <br>
        I want to merge and summarise data from multiple datasets, <br>
        so that I can create comprehensive reports without manual data cleaning.<br>
        <div class="acceptance-criteria">
        <strong>Acceptance Criteria:</strong>
        <ol>
            <li>User can upload multiple datasets</li>
            <li>System provides options for merging datasets based on common fields</li>
            <li>User can select aggregation methods (sum, average, count, etc.)</li>
            <li>System generates a new dataset with merged and aggregated data</li>
            <li>User can prompt to download the ideal dataset containing merged and summarised data</li>
        </ol>
        </div>
        </div>
        For more reference, please visit: 
        <a href="https://your-help-link.com" target="_blank">this link</a>
        For dummy data: 
        <a href="https://www.kaggle.com/datasets/tamsnd/adidas-webstore-shoe-data?resource=download&select=shoes_fact.csv" target="_blank">Kaggle link</a>
        </p>''',
            unsafe_allow_html=True
        )

        if st.button("Clear Cache", key="clear_cache_sidebar"):
            st.session_state.clear()
            st.rerun()

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload Data files",
        type=list(file_formats.keys()),
        help="Various File formats are Supported",
        on_change=clear_submit,
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.warning("Please upload at least one file to continue.")
        st.stop()

    # Load all dataframes
    dataframes = {}
    for uploaded_file in uploaded_files:
        df = load_data(uploaded_file)
        if df is not None:
            name = os.path.splitext(uploaded_file.name)[0]
            dataframes[name] = df
            st.success(f"Loaded {uploaded_file.name}")

    if not dataframes:
        st.error("No valid dataframes were loaded.")
        st.stop()

    # Display dataframe information
    st.write("### Loaded Data Information")
    for name, df in dataframes.items():
        with st.expander(f"Preview {name}"):
            st.write(f"Shape: {df.shape}")
            st.write(df.head())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What would you like to know about the data?"):
        context_prompt = generate_context_prompt(dataframes)
        full_prompt = context_prompt + prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key
        )

        # Combine all dataframes into one with proper null handling
        combined_df = combine_dataframes(dataframes)
        
        if combined_df.empty:
            st.error("Failed to combine dataframes.")
            st.stop()

        # Create a pandas dataframe agent with the combined dataframe
        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            combined_df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            memory=memory  # Add memory here
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = pandas_df_agent.run(full_prompt, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()