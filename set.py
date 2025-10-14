from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Dictionary of Bible names with weights (Samuel has highest weight)
bible_name_weights = {
    "Samuel": 100,
    "David": 60,
    "Moses": 55,
    "Abraham": 50,
    "Isaac": 45,
    "Jacob": 45,
    "Joseph": 40,
    "Joshua": 40,
    "Aaron": 35,
    "Elijah": 35,
    "Elisha": 30,
    "Noah": 30,
    "Daniel": 30,
    "Solomon": 30,
    "Saul": 25,
    "Peter": 25,
    "Paul": 25,
    "John": 25,
    "James": 25,
    "Andrew": 20,
    "Philip": 20,
    "Thomas": 20,
    "Bartholomew": 20,
    "Matthew": 20,
    "Simon": 20,
    "Jude": 20,
    "Barnabas": 20,
    "Stephen": 20,
    "Timothy": 20,
    "Titus": 20,
    "Zacharias": 15,
    "Elizabeth": 15,
    "Mary": 15,
    "Martha": 15,
    "Lazarus": 15,
    "Job": 15,
    "Ezra": 15,
    "Nehemiah": 15,
    "Esther": 15,
    "Ruth": 15,
    "Boaz": 15,
    "Gideon": 15,
    "Deborah": 15,
    "Hannah": 15,
    "Rebekah": 15,
    "Leah": 15,
    "Rachel": 15,
    "Ezekiel": 15,
    "Isaiah": 15
}

# Generate word cloud from dictionary
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Purples',
    max_words=50
).generate_from_frequencies(bible_name_weights)

# Save as JPEG
wordcloud.to_file("bible_names_weighted_wordcloud.jpeg")

# Display the image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
