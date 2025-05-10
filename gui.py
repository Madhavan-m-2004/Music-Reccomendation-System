import tkinter as tk
from music_recommender import load_data, create_user_song_matrix, get_collab_recommendations, get_content_recommendations

df = load_data("music_data.csv")
user_song_matrix = create_user_song_matrix(df)

def show_recommendations():
    try:
        user_id = int(user_id_entry.get())
        song_name = song_entry.get()
        
        collab = get_collab_recommendations(user_id, user_song_matrix)
        content = get_content_recommendations(song_name, df)
        
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, f"Collaborative Filtering:\n{collab}\n\n")
        result_text.insert(tk.END, f"Content-Based Filtering:\n{content}")
    except Exception as e:
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, f"Error: {e}")

root = tk.Tk()
root.title("Music Recommendation System")

tk.Label(root, text="Enter User ID:").pack()
user_id_entry = tk.Entry(root)
user_id_entry.pack()

tk.Label(root, text="Enter a Song Name:").pack()
song_entry = tk.Entry(root)
song_entry.pack()

tk.Button(root, text="Get Recommendations", command=show_recommendations).pack()

result_text = tk.Text(root, height=10, width=60)
result_text.pack()

root.mainloop()
