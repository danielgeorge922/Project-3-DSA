import csv
import sys
import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from scipy.spatial import distance

"""
TODO: Show top 10 songs and top 10 genres per user
TODO: Determine a similarity score in musical taste between users
TODO: Use NetworkX to visualize the graph
TODO: Write a control similar to the following:

Welcome to SpotiMatch
----------------------------------------
    1. Song Matcher
        --> Input data file
        --> List the amount of songs you want to put into the matcher (1 - 10)
    2. Top 10 Songs
        --> Input data file
    3. Top 10 Genres
        --> Input data file
    4. User Similarity
        --> Input # users to compare in musical taste
    5. Graph Visualization
        --> Input data files

NOTE: graph_similarity_score's bfs is implemented in such a way that it returns the first 4 entries of the graph,
and the song that was inputted by the user in first position. 
"""


class HashTable:
    def __init__(self, size=1024):
        self.size = size
        self.table = [None] * size
        self.items_count = 0

    def hash_function(self, key):
        hash_code = 0
        for char in key:
            hash_code = (hash_code * 31 + ord(char)) % self.size
        return hash_code

    def insert(self, key, value):
        index = self.hash_function(key)
        if self.table[index] is None:
            self.table[index] = [(key, value)]
        else:
            for i, (item_key, item_value) in enumerate(self.table[index]):
                if item_key == key:
                    # Assuming value is a dictionary with a 'time' key.
                    # Update existing entry if identical key found
                    item_value['time'] *= 2  # Doubles the 'time' value
                    self.table[index][i] = (key, item_value)
                    return
            self.table[index].append((key, value))
        self.items_count += 1

    def search(self, key):
        index = self.hash_function(key)
        if self.table[index] is not None:
            for item_key, value in self.table[index]:
                if item_key == key:
                    return value
        return None

    def resize(self):
        old_table = self.table
        self.size *= 2
        self.table = [None] * self.size
        for bucket in old_table:
            if bucket:
                for key, value in bucket:
                    self.insert(key, value)

class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_song(self, song_name, song_metrics):
        if song_name in self.adj_list:
            # If the song already exists, double the 'time' metric
            if 'time' in self.adj_list[song_name]["metrics"]:
                self.adj_list[song_name]["metrics"]["time"] *= 2
            return
        
        self.adj_list[song_name] = {"metrics": song_metrics, "neighbors": []}

    def add_edge(self, song1, song2):
        if song1 in self.adj_list and song2 in self.adj_list:
            metrics1 = np.array([self.adj_list[song1]["metrics"][key] for key in ["danceability", "energy"]])
            metrics2 = np.array([self.adj_list[song2]["metrics"][key] for key in ["danceability", "energy"]])
            distance = np.linalg.norm(metrics1 - metrics2)
            self.adj_list[song1]["neighbors"].append((song2, 1 / (1 + distance)))
            self.adj_list[song2]["neighbors"].append((song1, 1 / (1 + distance)))

    def get_similar_songs(self, song_name):
        if song_name not in self.adj_list:
            return []
        
        visited = set()
        pq = []
        heapq.heappush(pq, (0, song_name))
        similar_songs = []

        while pq:
            dist, song = heapq.heappop(pq)
            if song not in visited:
                visited.add(song)
                similar_songs.append((song, dist))
                for neighbor, weight in self.adj_list[song]["neighbors"]:
                    if neighbor not in visited:
                        heapq.heappush(pq, (dist + weight, neighbor))

        return similar_songs[:5]  # Return top 5 similar songs

def process_csv(file_name, container, is_graph=False):
    # Define the fields from the CSV to be used in the HashTable and Graph
    relevant_fields_hash = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
        'duration_ms', 'time_signature'
    ]

    relevant_fields_graph = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
        'duration_ms', 'time_signature'
    ]

    start_time = time.time()  # Start timing
    try:
        with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    song_name = row['song_name']
                    if is_graph:
                        song_metrics = {key: float(row[key]) for key in relevant_fields_graph if key in row}
                        container.add_song(song_name, song_metrics)
                    else:
                        song_metrics = {key: float(row[key]) for key in relevant_fields_hash if key in row}
                        container.insert(song_name, song_metrics)
                except ValueError as ve:
                    print(f"Skipping row due to error: {ve}")
                except KeyError as ke:
                    print(f"Missing key {ke} in data; this row will be skipped.")
    except Exception as e:
        print(f"Failed to process file {file_name}: {e}")
    finally:
        end_time = time.time()
        print(f"Time to process {file_name} into {'Graph' if is_graph else 'HashTable'}: {end_time - start_time:.2f} seconds")

def bfs(graph, start_song, songs_list):
    # Initialize distance dictionary to store distances from start_song to other songs
    distance = {song: float('inf') for song in graph.adj_list}
    distance[start_song] = 0

    # Initialize a queue for BFS traversal
    queue = deque([start_song])

    # Perform BFS traversal
    while queue:
        current_song = queue.popleft()
        if current_song in songs_list:
            songs_list.remove(current_song)  # Remove user-input songs from consideration
        for neighbor, _ in graph.adj_list[current_song]["neighbors"]:
            if distance[neighbor] == float('inf'):
                distance[neighbor] = distance[current_song] + 1
                queue.append(neighbor)

    # Sort the songs by their distances from the start song and return the top 5
    closest_songs = sorted(distance.items(), key=lambda x: x[1])[:5]
    return closest_songs


def graph_song_similarity(songs_list, graph):
    start_time = time.time()

    # Initialize the closest songs list
    closest_songs = []

    # Perform BFS for each user-input song
    for start_song in songs_list:
        similar_songs = bfs(graph, start_song, songs_list)
        closest_songs.extend(similar_songs)
        
        # Sort the closest songs and keep only top 5
        closest_songs.sort(key=lambda x: x[1])
        closest_songs = closest_songs[:5]

    end_time = time.time()
    time_taken_ms = (end_time - start_time) * 1000

    # Print the top 5 similar songs
    print("-" * 40)
    print("\nTop 5 songs similar to your listening history:\n")
    for i, (song, _) in enumerate(closest_songs, start=1):
        print(f"{i}. {song}")

    print(f"\nTime taken with BFS implementation: {time_taken_ms:.2f} ms\n")
    print("-" * 40)

def hash_table_similarity(songs_list, hash_table):
    start_time = time.time()
    song_scores = []
    
    for song in songs_list:
        result = hash_table.search(song)  # Get result from hash table
        if result:  # Check if the result is not empty
            try:
                # Using 'duration_ms' as the metric
                song_scores.append(result['duration_ms'])  # Adjust to the correct metric key
            except KeyError as e:
                print(f"Error: Key '{e}' not found in result for song '{song}'.")
        else:
            print(f"No data found for {song}")

    if not song_scores:
        print("No valid song data to process.")
        return

    average_of_songs = sum(song_scores) / len(song_scores)
    closest_songs = []
    
    for bucket in hash_table.table:
        if bucket:  # Check if the bucket is not empty
            for song, values in bucket:
                try:
                    song_metric = values['duration_ms']  # Using 'duration_ms' as the metric
                    difference = abs(song_metric - average_of_songs)
                    closest_songs.append((song, difference))
                except KeyError as e:
                    print(f"Error: Key '{e}' not found while processing song '{song}'.")

    # Sort by the smallest difference and get the top 5
    closest_songs.sort(key=lambda x: x[1])
    top_5_closest = closest_songs[:5]

    end_time = time.time()
    time_taken_ms = (end_time - start_time) * 1000

    print("-" * 40)
    print("\n5 Songs You Might Like Based off Listening History\n")
    for i, (song, _) in enumerate(top_5_closest, start=1):
        print(f"{i}. {song}")
    print(f"\nTime taken with Hash Table implementation: {time_taken_ms:.2f} ms\n")

def normalize_title(title):
    """Normalize the title to lower case for uniform comparison."""
    return title.lower().strip()

def top_10_songs(container, is_graph=False):
    heap = []
    start_time = time.time()

    if is_graph:
        # Example of handling graph logic; assuming each song in graph.adj_list has a 'popularity' metric
        for song, data in container.adj_list.items():
            popularity = data['metrics'].get('popularity', 0)  # Replace 'popularity' with your actual metric
            if len(heap) < 10:
                heapq.heappush(heap, (popularity, song))
            elif popularity > heap[0][0]:
                heapq.heappushpop(heap, (popularity, song))
    else:
        # Assuming hash table buckets contain song metrics including 'popularity'
        for bucket in container.table:
            if bucket:
                for song, metrics in bucket:
                    popularity = metrics.get('popularity', 0)  # Replace 'popularity' with your actual metric
                    if len(heap) < 10:
                        heapq.heappush(heap, (popularity, song))
                    elif popularity > heap[0][0]:
                        heapq.heappushpop(heap, (popularity, song))

    end_time = time.time()
    time_taken_ms = (end_time - start_time) * 1000

    print("Top 10 Songs:")
    sorted_songs = sorted(heap, reverse=True, key=lambda x: x[0])  # Sort descending based on the metric
    for i, (metric, song) in enumerate(sorted_songs, start=1):
        print(f"{i}. {song} ")

    print(f"Time taken with {'Graph' if is_graph else 'HashTable'} implementation: {time_taken_ms:.2f} ms\n")
    
def top_10_genres(file_name):
    genre_count = {}
    start_time = time.time()

    with open(file_name, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            genre = row['genre'].strip()  # Assuming the genre is not a list and just a string
            if genre:
                genre_count[genre] = genre_count.get(genre, 0) + 1

    # Extract the top 10 genres
    top_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)[:10]
    
    end_time = time.time()
    time_taken_ms = (end_time - start_time) * 1000
    
    print("Top 10 Genres:")
    for i, (genre, count) in enumerate(top_genres, start=1):
        print(f"{i}. {genre}: {count}")
    
    print(f"Time taken: {time_taken_ms:.2f} ms\n") 
    
def load_song_data(file_name):
    """Loads song data from a CSV file into a list of dictionaries."""
    with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        return [row for row in csv_reader]

def recommend_similar_songs(data, song_name, num_recommendations=5):
    """Recommends songs based on Euclidean distance of song features."""
    song_features = {}
    feature_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    for row in data:
        features = [float(row[key]) for key in feature_keys if key in row]
        song_features[row['song_name']] = np.array(features)
    if song_name not in song_features:
        print(f"The song '{song_name}' was not found in the dataset.")
        return
    distances = [(name, distance.euclidean(song_features[song_name], features)) for name, features in song_features.items() if name != song_name]
    distances.sort(key=lambda x: x[1])
    print(f"Songs similar to '{song_name}':")
    for i, (name, dist) in enumerate(distances[:num_recommendations], start=1):
        print(f"{i}: {name} (Distance: {dist:.2f})")

def mood_based_playlist(data, mood, num_songs=10):
    """Creates a playlist based on a specified mood."""
    moods = {
        'happy': {'valence': (0.5, 1.0), 'energy': (0.5, 1.0)},
        'sad': {'valence': (0.0, 0.5), 'energy': (0.0, 0.5)},
        'calm': {'valence': (0.0, 0.5), 'energy': (0.0, 0.5)},
        'energetic': {'valence': (0.5, 1.0), 'energy': (0.5, 1.0)}
    }
    playlist = []
    for row in data:
        if all(moods[mood][key][0] <= float(row[key]) <= moods[mood][key][1] for key in moods[mood] if key in row):
            playlist.append(row['song_name'])
            if len(playlist) == num_songs:
                break
    print(f"Playlist for '{mood}' mood:")
    for i, song in enumerate(playlist, start=1):
        print(f"{i}. {song}")

def visualize_top_genres(data, top_n=10):
    """Visualizes the top genres based on their count."""
    sns.set(style="whitegrid")
    
    # Count the occurrences of each genre
    genre_counts = {}
    for row in data:
        genre = row['genre'].strip()  # Ensure we strip any leading/trailing whitespace
        if genre:
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1
    
    # Sort genres by count and select the top_n genres
    top_genres = sorted(genre_counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    genres, counts = zip(*top_genres)  # This unpacks the top genres and their counts into two lists
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(genres))
    plt.title(f'Top {top_n} Genres by Count')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <data_file.csv>")
        sys.exit(1)

    file_name = sys.argv[1]
    if not file_name.startswith("DATA/"):
        file_name = "DATA/" + file_name

    data = load_song_data(file_name)
    
    # Create instances of Graph and HashTable
    graph = Graph()
    hash_table = HashTable(45000)

    # Process the CSV file to populate graph and hash table
    print("\nProcessing data...\n")
    start_time = time.time()
    process_csv(file_name, hash_table)
    process_csv(file_name, graph, is_graph=True)
    end_time = time.time()
    print(f"Data processing completed in {end_time - start_time:.2f} seconds\n")

    while True:
        print("-" * 40)
        print("*" * 8 + " Welcome to SpotiMatch " + "*" * 9)
        print("-" * 40)
        print("   MENU:")
        print("1. Song Matcher")
        print("2. Top 10 Songs")
        print("3. Top 10 Genres")
        print("4. Recommend Similar Songs")
        print("5. Generate Mood-Based Playlist")
        print("6. Visualize Features")
        print("0. Exit")
        print("-" * 40)

        option = input("Please select an option from the menu above: [0 to quit] ")
        if not option.isdigit() or not (0 <= int(option) <= 6):
            print("Please enter a valid option.")
            continue
        option = int(option)

        if option == 0:
            print("Goodbye!")
            break

        if option == 1:
            try:
                songs_amount = int(input("List the amount of songs you want to put into the matcher (1 - 10): "))
                if not 1 <= songs_amount <= 10:
                    print("The number of songs must be between 1 and 10.")
                    continue
            except ValueError:
                print("Invalid number of songs entered.")
                continue

            songs_list = []
            i = 0
            while len(songs_list) < songs_amount:
                song_name = input(f'Song {i+1}: ').strip()
                normalized_input = normalize_title(song_name)
                found_in_data = False
                for key in graph.adj_list:
                    if normalized_input == normalize_title(key):
                        songs_list.append(key)
                        print(f"Added '{key}' based on your input '{song_name}'.")
                        found_in_data = True
                        print()
                        break

                if not found_in_data:
                    result = hash_table.search(normalized_input)
                    if result:
                        songs_list.append(song_name)
                        print("Successful")
                    else:
                        print(f"Song '{song_name}' not found. Please try another.")
                        continue  # Skip incrementing the counter if song not found

                i += 1  # Increment the counter only if a song is successfully added

            if songs_list:
                hash_table_similarity(songs_list, hash_table)
                graph_song_similarity(songs_list, graph)
                print()
            else:
                print("No valid songs were inputted for processing.")

        elif option == 2:
            print("\nFetching top 10 songs/genres...\n")
            start_time = time.time()
            top_10_songs(graph, is_graph=(option == 2))
            end_time = time.time()
            print(f"Top 10 list generated in {end_time - start_time:.2f} seconds\n")
        
        elif option == 3:
            # The file_name variable should already contain the CSV file path
            print("\nFetching top 10 genres...\n")
            top_10_genres(file_name)  # Call the function with the CSV file path

        elif option == 4:
            song_name = input("Enter the song name to find similar songs: ")
            recommend_similar_songs(data, song_name)
            
        elif option == 5:
            mood = input("Enter a mood (happy, sad, calm, energetic): ")
            mood_based_playlist(data, mood)
            
        elif option == 6:
            visualize_top_genres(data, top_n=10) 

        input("Press Enter to continue...")

if __name__ == '__main__':
    main()
    