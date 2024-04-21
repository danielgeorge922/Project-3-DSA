import csv
import sys
import time
import heapq
import numpy as np
import sys
from collections import deque

class HashTable:
    def __init__(self, size=1024):
        self.size = size
        self.table = [None] * size
        self.items_count = 0

    def hash_function(self, key):
        # Simple hash function to distribute keys more uniformly
        hash_code = 0
        for char in key:
            hash_code = (hash_code * 31 + ord(char)) % self.size
        return hash_code

    def insert(self, key, value):
        index = self.hash_function(key)
        if self.table[index] is None:
            self.table[index] = [(key, value)]
        else:
            for i, (item_key, _) in enumerate(self.table[index]):
                if item_key == key:
                    # Update existing entry if identical key found
                    self.table[index][i] = (key, value)
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
            print(f"'{song_name}' already exists. Consider updating instead of adding.")
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
    relevant_fields_hash = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                       'duration_ms', 'time_signature']
    relevant_fields_graph = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                       'duration_ms', 'time_signature', 'genre']
    try:
        with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    song_name = row['song_name']
                    if is_graph:
                        song_metrics = {key: (float(row[key]) if key != 'genre' else row[key]) for key in relevant_fields_graph if key in row}
                        container.add_song(song_name, song_metrics)
                    else:
                        song_metrics = [float(row[key]) for key in relevant_fields_hash if key in row]
                        container.insert(song_name, song_metrics)
                except ValueError as ve:
                    print(f"Skipping row due to error: {ve}")
                except KeyError as ke:
                    print(f"Missing key {ke} in data; this row will be skipped.")
    except Exception as e:
        print(f"Failed to process file {file_name}: {e}")

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

def graph_similarity(songs_list, graph):
    start_time = time.time()

    # Initialize closest songs list
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
    print("Top 5 songs similar to your listening history:")
    for i, (song, _) in enumerate(closest_songs, start=1):
        print(f"{i}. {song}")

    print(f"Time taken with BFS implementation: {time_taken_ms:.2f} ms")

def hash_table_similarity(songs_list, hash_table):
    start_time = time.time()
    song_scores = []
    
    for song in songs_list:
        result = hash_table.search(song)  # Get result from hash table
        if result:  # Check if the result is not empty
            song_scores.append(result[0])  # Assuming result[0] is the metric you are interested in
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
                    song_metric = values[0]  # Assuming values[0] is the metric
                    difference = abs(song_metric - average_of_songs)
                    closest_songs.append((song, difference))
                except (TypeError, IndexError, ValueError) as e:
                    print(f"Error processing song {song}: {e}")

    # Sort by the smallest difference and get the top 5
    closest_songs.sort(key=lambda x: x[1])
    top_5_closest = closest_songs[:5]

    end_time = time.time()  # End the timer
    time_taken_ms = (end_time - start_time) * 1000  # Calculate the time taken in milliseconds

    print("-" * 40)
    print("5 Songs You Might Like Based off Listening History")
    for i, (song, _) in enumerate(top_5_closest, start=1):
        print(f"{i}. {song}")
    print(f"Time taken with Hash Table implementation: {time_taken_ms:.2f} ms")
    
def normalize_title(title):
    """Normalize the title to lower case for uniform comparison."""
    return title.lower().strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        return
    
    file_name = sys.argv[1]
    if not file_name.startswith("DATA/"):
        file_name = "DATA/" + file_name

    # Create instances of Graph and HashTable
    graph = Graph()
    hash_table = HashTable(45000)
    
    # Process the CSV file to populate graph and hash table
    process_csv(file_name, graph, is_graph=True)
    process_csv(file_name, hash_table)

    print("Welcome to SpotiMatch")
    print("-" * 40)
    try:
        songs_amount = int(input("List the amount of songs you want to put into the matcher (1 - 10): "))
        if not 1 <= songs_amount <= 10:
            print("The number of songs must be between 1 and 10.")
            return
    except ValueError:
        print("Invalid number of songs.")
        return

    songs_list = []
    for i in range(songs_amount):
        song_name = input(f'Song {i+1}: ').strip()
        normalized_input = normalize_title(song_name)
        found = False
        # Search in graph and hash table
        for key in graph.adj_list:
            if normalized_input == normalize_title(key):
                songs_list.append(key)
                print(f"Added '{key}' based on your input '{song_name}'.")
                found = True
                break

        if not found:
            result = hash_table.search(normalized_input)
            if result:
                songs_list.append(song_name)
                print("Successful")
            else:
                print(f"Song '{song_name}' not found. Please try another.")

    if songs_list:
        hash_table_similarity(songs_list, hash_table)
        graph_similarity(songs_list, graph)
    else:
        print("No valid songs were inputted for processing.")

if __name__ == '__main__':
    main()
    






















































"""
def process_csv_for_graph(file_name, graph):
    with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            # Extract song name from the "song_name" column
            song_name = row['song_name']
            # Extract song metrics
            song_metrics = {
                "danceability": float(row['danceability']),
                "energy": float(row['energy']),
                "key": int(row['key']),
                "loudness": float(row['loudness']),
                "mode": int(row['mode']),
                "speechiness": float(row['speechiness']),
                "acousticness": float(row['acousticness']),
                "instrumentalness": float(row['instrumentalness']),
                "liveness": float(row['liveness']),
                "valence": float(row['valence']),
                "tempo": float(row['tempo']),
                "duration_ms": int(row['duration_ms']),
                "time_signature": int(row['time_signature']),
                "genre": row['genre']
            }
            # Add the song and its metrics to the graph
            graph.add_song(song_name, song_metrics)

def process_csv_for_hash_table(file_name, hash_table):
    with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            # Extract song name from the "song_name" column
            song_name = row['song_name']

            # Extract other data from specified columns and create a list
            song_metrics = [
                float(row['danceability']),
                float(row['energy']),
                int(row['key']),
                float(row['loudness']),
                int(row['mode']),
                float(row['speechiness']),
                float(row['acousticness']),
                float(row['instrumentalness']),
                float(row['liveness']),
                float(row['valence']),
                float(row['tempo']),
                int(row['duration_ms']),
                int(row['time_signature']),
                row['genre']
            ]

            # Insert song into the hash table
            hash_table.insert(song_name, song_metrics)
 """

# def handle_graph(songs_amount):
#     graph = Graph()
#     process_csv_for_graph('genres_v2.csv', graph)
#     i = 1
#     songs_list = []
#     print("-" * 40)
#     while songs_amount > 0:
#         song_name = input(f'Song {i}: ')
#         if song_name in graph.adj_list:
#             songs_list.append(song_name)
#             print("Successful")
#         else:
#             print(f"Song '{song_name}' not found in the graph")
#             continue
#         songs_amount = songs_amount - 1
#         i = i + 1
#     graph_similarity(songs_list,graph)
#
# def handle_hashtable(songs_amount):
#     hash_table = HashTable(45000)  # Initialize a hash table with size 45000
#     process_csv_for_hash_table('genres_v2.csv', hash_table)
#
#     i = 1
#     songs_list = []
#     while songs_amount > 0:
#         song_name = input(f'Song {i}: ')
#
#         found_metrics, time_taken_ms = hash_table.search(song_name)
#         if found_metrics:
#             # print(f"Song Metrics for '{song_name}' (HASH TABLE):", found_metrics)
#             songs_list.append(song_name)
#             print("Successful")
#         else:
#             print(f"Song '{song_name}' not found")
#             continue
#         songs_amount -= 1
#         i += 1
#
#     hash_table_similarity(songs_list,hash_table)

