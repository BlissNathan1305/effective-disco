from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Optional: Load a custom shape mask (e.g., gear shape)
# gear_mask = np.array(Image.open("gear_shape.png"))  # Uncomment if you have a shape image

# Engineering terms
engineering_terms = [
    "thermodynamics", "fluid", "mechanics", "circuit", "voltage", "current", "resistance", "inductor", "capacitor",
    "transistor", "semiconductor", "algorithm", "data", "structure", "compiler", "network", "protocol", "signal",
    "control", "robotics", "automation", "design", "CAD", "stress", "strain", "load", "beam", "torque", "gear",
    "motor", "sensor", "actuator", "feedback", "system", "modeling", "simulation", "analysis", "efficiency",
    "power", "energy", "heat", "transfer", "material", "composite", "welding", "fabrication", "manufacturing",
    "process", "quality", "safety", "engineering"
]

# Combine terms into a single string
text = " ".join(engineering_terms)

# Define custom color function
def custom_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return "rgb(0, 102, 204)"  # Deep blue

# Create word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    color_func=custom_color,
    max_words=50,
    contour_color="black",
    contour_width=1,
    # mask=gear_mask,  # Uncomment if using a shape mask
    font_path=None  # You can specify a custom font path here
).generate(text)

# Save as JPEG
wordcloud.to_file("custom_engineering_wordcloud.jpeg")

# Display the image
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
