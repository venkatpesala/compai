import camelot

# Path to your PDF file
pdf_path = 'pdfs/MSFT-DEF 14A 2023 (1).pdf'

# Extract tables using Camelot
tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')  # or flavor='lattice'

# Iterate through tables and save them to CSV
for i, table in enumerate(tables):
    table.to_csv(f'table_{i}.csv')
    print(f"Extracted table {i} and saved as CSV.")
