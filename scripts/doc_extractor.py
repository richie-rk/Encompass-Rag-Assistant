import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import csv
import re
from collections import deque
from rich import print

def crawl_documentation(start_url, output_file="documentation_data.csv", delay=1):
    """
    Crawl documentation starting from a home page URL, extract all unique URLs,
    and scrape content from each page.

    Args:
        start_url (str): The starting URL of the documentation
        output_file (str): File to save the scraped data
        delay (int): Delay between requests in seconds
    """
    parsed_url = urlparse(start_url)
    base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

    visited_urls = set()  # To track visited URLs
    url_queue = deque([start_url])  # Queue of URLs to visit
    scraped_data = []  # Store the scraped content

    print(f"Starting crawl from: {start_url}")
    print(f"Base domain: {base_domain}")

    while url_queue:
        current_url = url_queue.popleft()

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)

        print(f"Processing: {current_url}")

        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.title.text.strip() if soup.title else "No Title"

            content_selectors = ['main', 'article', '.content', '.documentation', '#content', '.main-content']
            content = ""
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content = content_element.get_text(separator=' ', strip=True)
                    break

            if not content:
                content = soup.body.get_text(separator=' ', strip=True) if soup.body else "No content"

            scraped_data.append({
                'url': current_url,
                'title': title,
                'content': content
            })

            links = soup.find_all('a')
            for link in links:
                href = link.get('href')

                if not href or href.startswith(('#', 'javascript:')):
                    continue

                absolute_url = urljoin(current_url, href)

                parsed_href = urlparse(absolute_url)
                href_domain = f"{parsed_href.scheme}://{parsed_href.netloc}"

                if href_domain == base_domain and absolute_url not in visited_urls:
                    url_queue.append(absolute_url)

            # To avoid HTTP 429 (Too Many Requests) errors and respect the website by waiting between requests
            time.sleep(delay)

        except Exception as e:
            print(f"Error processing {current_url}: {e}")

    save_to_csv(scraped_data, output_file)

    print(f"Crawl completed. Processed {len(visited_urls)} unique URLs.")
    return scraped_data


def save_to_csv(data, filename):
    """Save the scraped data to a CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'title', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item in data:
            writer.writerow(item)

    print(f"Data saved to {filename}")


if __name__ == "__main__":
    documentation_url = "https://developer.icemortgagetechnology.com/developer-connect/docs/welcome"
    crawl_documentation(documentation_url)
