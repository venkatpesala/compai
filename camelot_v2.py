import camelot
import pdfplumber
import os
import re

# Path to the directory containing the PDF
pdf_path = 'pdfs/MSFT-DEF 14A 2023 (1).pdf'
output_dir = 'extracted_tables'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def find_table_titles(pdf_path):
    """
    Extracts potential table titles from a PDF using pdfplumber.
    Titles are assumed to be text elements just before tables.
    """
    titles = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            # Use a regular expression or keyword matching to find table titles
            # This regex looks for common words like "Table" or "Summary" to find titles
            matches = re.findall(r'((?:Summary|Table|Outstanding|Compensation|Awards).+)', text, re.IGNORECASE)
            titles.extend(matches)
    return titles

def clean_table_title(title):
    """
    Cleans the table title by removing invalid characters and trimming.
    """
    title = re.sub(r'[\\/*?:"<>|]', '', title)  # Remove invalid characters for filenames
    return title.strip()

# Extract potential table titles from the PDF
table_titles = find_table_titles(pdf_path)

# Extract tables from the PDF using Camelot
tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')  # or 'lattice' if tables have clear grid lines

# If there are more tables than titles, we'll generate a generic name
for i, table in enumerate(tables):
    if i < len(table_titles):
        table_title = clean_table_title(table_titles[i])
    else:
        table_title = f"Table_{i+1}"

    # Save the table as CSV
    csv_filename = os.path.join(output_dir, f"{table_title}.csv")
    table.to_csv(csv_filename)

    print(f"Extracted and saved table {i + 1} as {csv_filename}")
