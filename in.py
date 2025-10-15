from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define word frequencies (Inemeno is boldest)
word_frequencies = {
    "Inemeno": 100,  # Highest frequency
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
    "Ekpededeme": 10,
    "Jelly tots": 10,
    "Moon pie": 10,
    "Cupcake": 10,
    "Most adorable": 10,
    "My Irreplaceable": 10,
    "My eternal partner": 10,
    "My pride": 10,
    "My incalculable blessings": 10,
    "My woman of God": 10,
    "My Girlfriend": 10,
    "My better half": 10,
    "Ife mi": 10,
    "My love bud": 10
}

# Create the word cloud for portrait 4K resolution
wordcloud = WordCloud(
    width=2160,
    height=3840,
    background_color='white',
    colormap='plasma',
    collocations=False
).generate_from_frequencies(word_frequencies)

# Set up the figure with matching dimensions and high DPI
plt.figure(figsize=(21.6, 38.4), dpi=100)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Export as high-quality JPEG
wordcloud.to_file("wordcloud_portrait_4k.jpeg")
print("Portrait 4K word cloud saved as 'wordcloud_portrait_4k.jpeg'")
