import os
from unstructured.partition.pdf import partition_pdf
import pandas as pd
import ssl
import urllib.request
from unstructured.partition.auto import partition

ssl._create_default_https_context = ssl._create_unverified_context


# ---------------------------- Configuration ---------------------------- #

# Path to the directory containing DEF-14 PDF files
PDF_DIR = 'pdfs'

# Path to the directory where extracted data will be saved
OUTPUT_DIR = 'extracted_data'

# List of tables and text sections to extract
REQUIRED_TABLES = [
    "Summary Compensation Table",
    "Grants of Plan-Based Awards",
    "Outstanding Equity Awards at June 30, 2023",
    "Option Exercises and Stock Vested",
    "Nonqualified Deferred Compensation",
    "Pay for Performance Table"
]

# Text sections to extract
REQUIRED_TEXTS = [
    "CEO Pay Ratio"
]

# ---------------------------------------------------------------------- #

def ensure_directories():
    """Ensure that the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_tables_and_text(pdf_path):
    """
    Extracts specified tables and text sections from a PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: A dictionary containing extracted tables and texts.
    """
    print(f"Processing: {pdf_path}")
    extracted_data = {
        "tables": {table: [] for table in REQUIRED_TABLES},
        "texts": {text: [] for text in REQUIRED_TEXTS}
    }

# 1st way

    

    elements = partition(filename=filename,
                        strategy='hi_res',
            )

    tables = [el for el in elements if el.category == "Table"]

    print(tables[0].text)
    print(tables[0].metadata.text_as_html)

    for element in elements:
        # Check if the element has 'type' and handle tables
        if hasattr(element, 'type') and element.type == 'table':
            # Attempt to associate the table with its preceding title
            table_title = previous_text.strip()
            if table_title in REQUIRED_TABLES:
                # Convert the table to a DataFrame
                try:
                    df = pd.DataFrame(element.cells)
                    # Assume first row as header
                    df.columns = df.iloc[0]
                    df = df[1:]
                    extracted_data["tables"][table_title].append(df)
                    print(f"Extracted table: {table_title}")
                except Exception as e:
                    print(f"Error extracting table '{table_title}': {e}")
        elif hasattr(element, 'type') and element.type == 'text':
            text_content = element.text.strip()
            # Check for required text sections
            for required_text in REQUIRED_TEXTS:
                if required_text.lower() in text_content.lower():
                    extracted_data["texts"][required_text].append(text_content)
                    print(f"Extracted text section: {required_text}")
            # Update previous_text for potential table titles
            previous_text = text_content
        else:
            # Handle cases where the element doesn't have 'type', like 'Header' or unknown types
            previous_text = ""
            print(f"Skipping element of type {type(element).__name__}, not text or table.")

    return extracted_data

def save_extracted_data(pdf_filename, data):
    """
    Saves the extracted tables and texts to the output directory.

    Args:
        pdf_filename (str): The name of the PDF file.
        data (dict): The extracted data containing tables and texts.
    """
    base_filename = os.path.splitext(pdf_filename)[0]

    # Save tables
    for table_name, tables in data["tables"].items():
        for idx, df in enumerate(tables, start=1):
            # Clean table name for filename
            clean_table_name = table_name.replace(" ", "_").replace("/", "_")
            csv_filename = f"{base_filename}_{clean_table_name}_{idx}.csv"
            csv_path = os.path.join(OUTPUT_DIR, csv_filename)
            try:
                df.to_csv(csv_path, index=False)
                print(f"Saved table to {csv_path}")
            except Exception as e:
                print(f"Error saving table '{table_name}' to CSV: {e}")

    # Save texts
    for text_name, texts in data["texts"].items():
        for idx, text in enumerate(texts, start=1):
            # Clean text name for filename
            clean_text_name = text_name.replace(" ", "_").replace("/", "_")
            txt_filename = f"{base_filename}_{clean_text_name}_{idx}.txt"
            txt_path = os.path.join(OUTPUT_DIR, txt_filename)
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved text to {txt_path}")
            except Exception as e:
                print(f"Error saving text '{text_name}' to TXT: {e}")

def main():
    """Main function to orchestrate extraction and saving."""
    ensure_directories()

    # Iterate over all PDF files in the PDF_DIR
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.lower().endswith('.pdf'):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            extracted_data = extract_tables_and_text(pdf_path)
            save_extracted_data(pdf_file, extracted_data)

    print("Extraction process completed.")

if __name__ == "__main__":
    main()
