# Fake-News-Detector

Description: 
==========================
This program employs a Multinomial Naive Bayes Classifier algorithm to predict whether a news article is "Fake News" or not.

After completing the Datacamp course, "Introduction to Natural Language Processing in Python", I was inspired to apply my newly acquired knowledge create this progam. The Datacamp course instructs how to build a Fake News classifier and tune the model for performance. After using this code for inspiration, I built my program to:
1) Allow users to test indiviudal articles for themselves. The Datacamp course code only permitted a user to test model performance on a separate test set.
2) Have web scraping functionality, allowing a user to test an article by submitting a URL.
3) Have GUI functionality. 




Structure of program
==========================
-- File: fakenews_gui.py -- The first half contains the machine learning section of the program. The second half contains the GUI section.

-- File: fakenews_scraper.py -- If a user submits a URL to the program rather than text to an article, a scraper is called to harvest article text from the given URL.




Acknowlegements: 
==========================
-- The following Datacamp file was used for training data (https://assets.datacamp.com/production/repositories/932/datasets/cd04303b8b2904d1025809dfb29076de696a1ffc/News%20articles.zip)

-- Link to Datacamp's "Introduction to Natural Language Processing in Python" (https://www.datacamp.com/courses/natural-language-processing-fundamentals-in-python)
