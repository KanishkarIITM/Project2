# /// script
# dependencies = [
#   "os",
#   "pandas",
#   "requests",
#   "seaborn",
#   "matplotlib",
#   "argparse",
#   "dotenv",
# ///

import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")


if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN is not set. Please set the token as an environment variable.")
    exit(1)

def llm_analysis(dataset_summary):
    dataset_summary_str = dataset_summary.head(10).to_string()
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json',
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that analyzes datasets and provides good and detailed insights."
            },
            {
                "role": "user",
                "content": f"The dataset summary: {dataset_summary_str}. Can you analyze it and provide insights?"
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        analysis_results = response.json()
        return analysis_results['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"Error querying AI proxy: {e}")
        return "AI analysis could not be performed due to an error."

def correlation_mapper(df):
    numeric_df = df.select_dtypes(include='number').dropna()

    if numeric_df.empty:
        print("Warning: No numerical columns available for correlation heatmap.")
        return None

    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)

    correlation_plot_path = "correlation_heatmap.png"
    plt.savefig(correlation_plot_path)
    plt.close()
    return correlation_plot_path

def gen_readme(df, analysis, correlation_plot_path):
    with open("README.md", "w") as file:
        file.write("## Data Summary\n")
        file.write(f"{df.describe()}\n")
        file.write("# Dataset Analysis Report\n")
        file.write("## Insights from AI Analysis\n")
        file.write(f"{analysis}\n")
        if correlation_plot_path:
            file.write("## Data Visualizations can be found in the directory...\n")
        else:
            file.write("## Data Visualizations\nNo numerical data available for a correlation heatmap.\n")

def read_csv_with_encodings(file_path):
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']
    for encoding in encodings_to_try:
        try:
            print(f"Trying to read the file with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            print(f"Failed to read the file with encoding: {encoding}")
    print("Error: Could not read the file with any of the tried encodings.")
    exit(1)

def main(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        return

    df = read_csv_with_encodings(file_path)

    correlation_plot_path = correlation_mapper(df)
    if correlation_plot_path:
        print(f"Correlation Heatmap saved at: {correlation_plot_path}")

    analysis_results = llm_analysis(df)

    gen_readme(df, analysis_results, correlation_plot_path)
    print("README.md file generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run dataset analysis')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')

    args = parser.parse_args()

    main(args.file_path)
