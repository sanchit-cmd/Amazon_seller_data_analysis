# Amazon Net Profit Calculator

Run the Streamlit app from the project directory:

```bash
uv run streamlit run main.py
```

Upload the four original Amazon reports:

- Orders export as a tab-separated `.txt` file
- Unified Transaction report as a `.csv` file with its 13 introductory rows
- Returns report as a tab-separated `.tsv` file
- SKU Master as an `.xlsx` file

The app matches returns by Order ID and SKU, calculates COGS for non-returned
items, adds packaging cost per unique order, and subtracts ad and other
expenses. Results can be downloaded as an Excel workbook with profit summary,
item details, and order summary sheets.