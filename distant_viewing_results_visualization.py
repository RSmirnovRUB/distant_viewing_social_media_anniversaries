import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# === Load data ===
df = pd.read_csv("/Users/rsmirnov/Desktop/distant_viewing_dataset.csv")

# === 1. Basic stats ===
print("Dataset size:", len(df))
print("Average brightness:", df["brightness"].mean())
print("Average color (R,G,B):", df[["avg_R", "avg_G", "avg_B"]].mean().values)

# === 2. Faces stats ===
face_fraction = (df["num_faces"] > 0).mean() * 100
print(f"Faces detected on {face_fraction:.1f}% of images")

# === 3. Brightness histogram ===
plt.figure(figsize=(6,4))
sns.histplot(df["brightness"], bins=20, kde=True)
plt.title("Brightness Distribution")
plt.xlabel("Brightness")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("brightness_hist.png")

# === 4. Average Color Plot ===
avg_color = df[["avg_R", "avg_G", "avg_B"]].mean().astype(int)
color_hex = "#{:02x}{:02x}{:02x}".format(*avg_color)

plt.figure(figsize=(2,2))
plt.imshow([[avg_color/255]])
plt.axis("off")
plt.title(f"Average Color {color_hex}")
plt.savefig("average_color.png")

# === 5. Top-10 Objects ===
all_objects = []
for x in df["objects_detected"].dropna():
    all_objects += [o.strip() for o in x.split(",") if o.strip()]

obj_counts = Counter(all_objects).most_common(10)
objects, counts = zip(*obj_counts)

plt.figure(figsize=(6,4))
sns.barplot(x=list(counts), y=list(objects), orient="h")
plt.title("Top-10 Detected Objects")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig("top_objects.png")

# === 6. Average Color Histograms ===
for channel, name in zip(["r", "g", "b"], ["Red", "Green", "Blue"]):
    cols = [c for c in df.columns if c.startswith(f"hist_{channel}_")]
    avg_hist = df[cols].mean()
    
    plt.figure(figsize=(5,3))
    plt.bar(range(8), avg_hist, color=name.lower())
    plt.xticks(range(8))
    plt.title(f"Average {name} Histogram")
    plt.xlabel("Bin (0-255 range)")
    plt.ylabel("Average count")
    plt.tight_layout()
    plt.savefig(f"hist_{channel}.png")

print("All plots saved as PNG files!")
