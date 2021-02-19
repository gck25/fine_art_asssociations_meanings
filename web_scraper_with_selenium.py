from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.request import urlopen

# url = "https://www.philamuseum.org/collections/permanent/82736.html"
url = "https://www.nationalgallery.org.uk/paintings/jan-jansz-treck-vanitas-still-life"
browser = webdriver.Firefox()
browser.get(url)
soup = BeautifulSoup(browser.page_source)
print(soup.get_text(), "lxml")
# for link in soup.find_all("a"):
#     print(link.get("href", None), link.get_text())

