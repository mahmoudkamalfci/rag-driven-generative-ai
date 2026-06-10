"""Pipeline 1: Fetch URLs, clean HTML content, save as text files."""
import os
import re
import requests
from bs4 import BeautifulSoup

URLS = [
    "https://github.com/VisDrone/VisDrone-Dataset",
    "https://paperswithcode.com/dataset/visdrone",
    "https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Zhu_VisDrone-DET2018_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ECCVW_2018_paper.pdf",
    "https://github.com/VisDrone/VisDrone2018-MOT-toolkit",
    "https://en.wikipedia.org/wiki/Object_detection",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    "https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle",
    "https://www.faa.gov/uas/",
    "https://www.tensorflow.org/",
    "https://pytorch.org/",
    "https://keras.io/",
    "https://arxiv.org/abs/1804.06985",
    "https://arxiv.org/abs/2202.11983",
    "https://motchallenge.net/",
    "http://www.cvlibs.net/datasets/kitti/",
    "https://www.dronedeploy.com/",
    "https://www.dji.com/",
    "https://arxiv.org/",
    "https://openaccess.thecvf.com/",
    "https://roboflow.com/",
    "https://www.kaggle.com/",
    "https://paperswithcode.com/",
    "https://github.com/"
]


def clean_text(content):
    """Remove references and unwanted characters."""
    content = re.sub(r'\[\d+\]', '', content)    # Remove references
    content = re.sub(r'[^\w\s\.]', '', content)   # Remove punctuation (except periods)
    return content


def fetch_and_clean(url):
    """Fetch URL and extract cleaned text content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Prioritize "mw-parser-output" but fall back to "content" class
        content = (
            soup.find('div', {'class': 'mw-parser-output'})
            or soup.find('div', {'id': 'content'})
        )
        if content is None:
            return None

        # Remove specific sections, including nested ones
        for section_title in ['References', 'Bibliography', 'External links',
                              'See also', 'Notes']:
            section = content.find('span', id=section_title)
            while section:
                for sib in section.parent.find_next_siblings():
                    sib.decompose()
                section.parent.decompose()
                section = content.find('span', id=section_title)

        text = content.get_text(separator=' ', strip=True)
        text = clean_text(text)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching {len(URLS)} URLs...")
    success_count = 0
    for url in URLS:
        article_name = url.split('/')[-1].replace('.html', '')
        if not article_name:
            article_name = url.rstrip('/').split('/')[-1]
        filename = os.path.join(output_dir, f"{article_name}.txt")

        print(f"  Fetching: {url}")
        clean_article_text = fetch_and_clean(url)
        if clean_article_text:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(clean_article_text)
            print(f"    → Saved ({len(clean_article_text)} chars)")
            success_count += 1
        else:
            print(f"    → Skipped (no content)")

    print(f"\nDone! {success_count}/{len(URLS)} saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
