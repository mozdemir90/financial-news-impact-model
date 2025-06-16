import requests
from bs4 import BeautifulSoup

url = "https://www.doviz.com/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

dolar = soup.find("span", {"data-socket-key": "USD"}).text
print("Anlık Dolar Kuru:", dolar)
