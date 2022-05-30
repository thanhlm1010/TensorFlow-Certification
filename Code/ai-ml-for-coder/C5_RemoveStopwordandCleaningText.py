from bs4 import BeautifulSoup
soup = BeautifulSoup(sentence)
sentence = soup.get_text()
stopwords = ["a", "about", "above", ... ,"yours", "yourself", "yourselves"]

