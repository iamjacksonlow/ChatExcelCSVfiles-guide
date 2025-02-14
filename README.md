# ChatExcelCSV - Merge & Summarize Data from CSV/Excel Files

## 📌 Overview
ChatExcelCSV is a Streamlit-based application that allows users to upload multiple datasets, merge them, and summarize data efficiently. The app integrates OpenAI's GPT models to analyze and interact with the uploaded data.
Medium article for full guide: - [How to Chat with Your Excel & CSV Files Using LangChain Agents and OpenAI’s GPT Models]([https://xxx/](https://medium.com/@iamjacksonlow/how-to-chat-with-your-excel-csv-files-using-langchain-agents-and-openais-gpt-models-ede8d1baf2c3))

## 🚀 Features
- Upload multiple CSV or Excel files.
- Merge datasets based on common fields.
- Select aggregation methods (sum, average, count, etc.).
- Generate a new dataset with merged and summarized data.
- Chat with the data using OpenAI's GPT models.
- Download the processed dataset.

## 🛠️ Installation & Setup

### 1️⃣ download or clone app.py
Download app.py and requirements.txt from the main page or clone this repos

### 2️⃣ Create a Virtual Environment (venv)
Ensure you have Python installed (preferably 3.8+). Then, create and activate a virtual environment:

#### On Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up OpenAI API Key
To use the GPT model, you need an OpenAI API key. Get yours from [OpenAI](https://openai.com/) and add it to the Streamlit app when prompted.

### 5️⃣ Run the Application
```sh
streamlit run app.py
```

## 📂 Supported File Formats
- CSV (`.csv`)
- Excel (`.xls`, `.xlsx`, `.xlsm`, `.xlsb`)

## 🛠️ How It Works
1. **Upload Data** - Upload multiple CSV/Excel files.
2. **Merge Data** - The system merges datasets based on common fields.
3. **Analyze Data** - Use GPT-powered chat to query and summarize your data.
4. **Download Processed Data** - Save the final dataset for further use.

## 🔗 Useful Links
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Kaggle Sample Data](https://www.kaggle.com/datasets/tamsnd/adidas-webstore-shoe-data?resource=download&select=shoes_fact.csv)

## 📜 License
MIT License - Feel free to use and modify this project!

---

💡 **Happy Data Merging! 🚀**

