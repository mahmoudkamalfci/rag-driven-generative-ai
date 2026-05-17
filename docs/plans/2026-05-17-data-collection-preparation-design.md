# Data Collection Preparation Script Design

## Objective
Convert the data collection and preparation steps from the Jupyter Notebook (`1_Data_collection_preparation.ipynb`) into a reusable, modular Python script (`data_collection_preparation.py`).

## Dependencies
- Package Manager: `uv`
- Environment: Use the existing `.venv`
- Packages: `beautifulsoup4`, `requests` (latest versions)

## Approach
We will use **Approach 2: Modular Functions** to ensure the code is clean, testable, and reusable for future RAG pipeline expansions.

## Core Components

1. **`clean_text(content: str) -> str`**
   - Helper function using regex to remove Wikipedia reference brackets (e.g., `[1]`, `[2]`).

2. **`fetch_and_clean_article(url: str) -> str`**
   - Uses `requests` to fetch the HTML content.
   - Parses the HTML with `BeautifulSoup`.
   - Strips out unwanted structural sections like "References", "Bibliography", "External links", and "See also".
   - Returns the cleaned text.

3. **`main()`**
   - Holds the target list of Wikipedia URLs.
   - Iterates through the URLs, calling the fetch/clean functions.
   - Includes progress printing for observability (e.g., `print(f"Processing {url}...")`).
   - Writes the aggregated, cleaned text into an output file: `llm.txt`.

4. **Execution Block**
   - The script will end with `if __name__ == "__main__": main()` to serve as the standard execution entry point.
