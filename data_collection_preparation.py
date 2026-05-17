import re
import requests
from bs4 import BeautifulSoup

def clean_text(content: str) -> str:
    """Remove references that usually appear as [1], [2], etc."""
    return re.sub(r'\[\d+\]', '', content)

def fetch_and_clean_article(url: str) -> str:
    """Fetch Wikipedia URL and extract cleaned article text."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main content
    content = soup.find('div', {'class': 'mw-parser-output'})
    if not content:
        return ""
        
    # Remove the bibliography/reference sections
    for section_title in ['References', 'Bibliography', 'External links', 'See also']:
        section = content.find('span', id=section_title)
        if section:
            for sib in section.parent.find_next_siblings():
                sib.decompose()
            section.parent.decompose()
            
    text = content.get_text(separator=' ', strip=True)
    return clean_text(text)
