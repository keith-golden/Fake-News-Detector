#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:26:29 2019

News Article Scraper

ArticleScraper class is used to go to a url and scrape text from a webpage

@author: Keith
"""
#################################################################
class ArticleScraper:
    """ 
    Given a url to a web article, it creates an object with url, title, and article (text content) attributes
    """
    def __init__(self, url):
        
        self.url = url
        
        import requests
        from bs4 import BeautifulSoup
        
        # Package the request, send the request and catch the response: r
        r = requests.get(url)
        
        # Parse the html content of the response object
        page_content = BeautifulSoup(r.content, "html.parser")
        
        # For all paragraphs, extract to text and add to article
        article = ""
        for i in range(len(page_content.find_all("p"))):    
            article = article + " " + page_content.find_all("p")[i].text
    
        # Strip leading space in article
        article = article.lstrip()
        
        # Set article attribute
        self.article = article
        
        # Now get the page title
        page_title = page_content.find_all("h1")[0].text
        self.title = page_title

      
    
    