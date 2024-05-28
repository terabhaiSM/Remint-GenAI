import requests
from bs4 import BeautifulSoup
import re 
import pandas as pd 

# URL of the webpage to scrape
url = 'https://www.reminthealth.com/blog/breaking-the-cycle-of-alcohol-addiction'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the div containing the blog content
    blog_content_div = soup.find('div', {'class': 'blog-detail-ul'})
    
    for tag in soup.find_all('p', class_='smal text-muted'):
        tag.decompose()
    # Check if the blog content div was found
    if blog_content_div:
        # Extract the blog content
        text = ''
        for element in blog_content_div.descendants:
            if element.name in ['p', 'h2']:
                text_content = element.get_text(strip=True)
                text_content = re.sub(r'\s+', ' ', text_content)  # Remove extra whitespace
                text += text_content + '\n'
    else:
        print('Blog content not found on the webpage.')
else:
    print(f'Failed to fetch the webpage. Status code: {response.status_code}')

# Save extracted text to a CSV file
with open('blog.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvfile.write(text)

print("CSV file 'blog.csv' has been created successfully.")
