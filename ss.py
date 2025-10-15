from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define word frequencies (higher frequency = bolder/larger word)
word_frequencies = {
    "Inemeno": 50,  # Highest frequency to make it boldest
    "Assian": 10,
    "Uyai": 10,
    "Ininghe": 10,
    "Edima": 10,
    "Ediye": 10,
    "Edidiong": 10,
    "Beryl": 10,
    "Fashionista": 10,
    "My pearl": 10,
    "My peace of mind": 10,
    "My Bundle of Joy incomparable": 10,
    "My Everyday Woman": 10,
    "My heartbeat": 10,
    "The wife of my youth": 10,
    "My safety net": 10,
    "My comforter": 10,
    "My Proverbs 31 Woman": 10,
    "Mummy GO": 10,
    "My worship Leader": 10,
    "My madam": 10,
    "My Love": 10,
    "LOML": 10,
    "My treasure": 10,
    "Mkpouto": 10,
    "Ekpededeme": 10
}

# Create the word cloud object
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='plasma',
    collocations=False
).generate_from_frequencies(word_frequencies)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Save the word cloud as a JPEG image
wordcloud.to_file("wordcloud_chart.jpeg")
print("Word cloud saved as 'wordcloud_chart.jpeg'")
