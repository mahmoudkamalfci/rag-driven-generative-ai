import re
import requests
from bs4 import BeautifulSoup

def clean_text(content: str) -> str:
    """Remove references that usually appear as [1], [2], etc."""
    return re.sub(r'\[\d+\]', '', content)

def fetch_and_clean_article(url: str) -> str:
    """Fetch Wikipedia URL and extract cleaned article text."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to fetch {url} - {e}")
        return ""
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

URLS = [
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/Apollo_program",
    "https://en.wikipedia.org/wiki/Hubble_Space_Telescope",
    "https://en.wikipedia.org/wiki/Mars_rover",
    "https://en.wikipedia.org/wiki/International_Space_Station",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/Juno_(spacecraft)",
    "https://en.wikipedia.org/wiki/Voyager_program",
    "https://en.wikipedia.org/wiki/Galileo_(spacecraft)",
    "https://en.wikipedia.org/wiki/Kepler_Space_Telescope",
    "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
    "https://en.wikipedia.org/wiki/Space_Shuttle",
    "https://en.wikipedia.org/wiki/Artemis_program",
    "https://en.wikipedia.org/wiki/Skylab",
    "https://en.wikipedia.org/wiki/NASA",
    "https://en.wikipedia.org/wiki/European_Space_Agency",
    "https://en.wikipedia.org/wiki/Ariane_(rocket_family)",
    "https://en.wikipedia.org/wiki/Spitzer_Space_Telescope",
    "https://en.wikipedia.org/wiki/New_Horizons",
    "https://en.wikipedia.org/wiki/Cassini%E2%80%93Huygens",
    "https://en.wikipedia.org/wiki/Curiosity_(rover)",
    "https://en.wikipedia.org/wiki/Perseverance_(rover)",
    "https://en.wikipedia.org/wiki/InSight",
    "https://en.wikipedia.org/wiki/OSIRIS-REx",
    "https://en.wikipedia.org/wiki/Parker_Solar_Probe",
    "https://en.wikipedia.org/wiki/BepiColombo",
    "https://en.wikipedia.org/wiki/Juice_(spacecraft)",
    "https://en.wikipedia.org/wiki/Solar_Orbiter",
    "https://en.wikipedia.org/wiki/CHEOPS_(satellite)",
    "https://en.wikipedia.org/wiki/Gaia_(spacecraft)"
]

def main():
    print(f"Starting data collection for {len(URLS)} articles...")
    with open('llm.txt', 'w', encoding='utf-8') as file:
        for url in URLS:
            print(f"Fetching: {url}")
            clean_article_text = fetch_and_clean_article(url)
            file.write(clean_article_text + '\n')
    print("Content written to llm.txt successfully.")

if __name__ == "__main__":
    main()
