import csv
import time
import numpy as np
from collections import deque

class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        if self.table[index] is None:
            self.table[index] = [(key, value)]
        else:
            self.table[index].append((key, value))

    def search(self, key):
        start_time = time.time()
        index = self.hash_function(key)
        if self.table[index] is not None:
            for item in self.table[index]:
                if item[0] == key:
                    end_time = time.time()
                    time_taken_ms = (end_time - start_time) * 1000
                    return item[1], time_taken_ms
        end_time = time.time()
        time_taken_ms = (end_time - start_time) * 1000
        return None, time_taken_ms

class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_song(self, song_name, song_metrics):
        if song_name not in self.adj_list:
            self.adj_list[song_name] = {"metrics": song_metrics, "neighbors": []}

    def add_edge(self, song1, song2):
        if song1 not in self.adj_list:
            self.adj_list[song1] = {"metrics": None, "neighbors": []}
        if song2 not in self.adj_list:
            self.adj_list[song2] = {"metrics": None, "neighbors": []}
        self.adj_list[song1]["neighbors"].append(song2)
        self.adj_list[song2]["neighbors"].append(song1)
 
def process_csv_for_hash_table(file_name, hash_table):
    relevant_fields = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                       'duration_ms', 'time_signature']
    try:
        with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    song_name = row['song_name']
                    song_metrics = [float(row[key]) for key in relevant_fields if key in row]
                    hash_table.insert(song_name, song_metrics)
                except ValueError as ve:
                    print(f"Skipping row due to error: {ve}")
                except KeyError as ke:
                    print(f"Missing key {ke} in data; this row will be skipped.")
    except Exception as e:
        print(f"Failed to process file {file_name}: {e}")


def process_csv_for_graph(file_name, graph):
    relevant_fields = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                       'duration_ms', 'time_signature', 'genre']
    try:
        with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    song_name = row['song_name']
                    song_metrics = {key: (float(row[key]) if key != 'genre' else row[key]) for key in relevant_fields if key in row}
                    graph.add_song(song_name, song_metrics)
                except ValueError as ve:
                    print(f"Skipping row due to error: {ve}")
                except KeyError as ke:
                    print(f"Missing key {ke} in data; this row will be skipped.")
    except Exception as e:
        print(f"Failed to process file {file_name}: {e}")

def graph_similarity(songs_list, graph):
    start_time = time.time()

    # Calculate the average song score
    total_danceability = sum(graph.adj_list[song]["metrics"]["danceability"] for song in songs_list)
    average_song_score = total_danceability / len(songs_list)

    # Initialize the closest songs list
    closest_songs = []

    # Perform BFS to find similar songs
    queue = deque(graph.adj_list.keys())
    visited = set()

    while queue:
        current_song = queue.popleft()
        current_metrics = graph.adj_list[current_song]["metrics"]
        difference = abs(current_metrics['danceability'] - average_song_score)
        closest_songs.append((current_song, difference))

        # Sort the closest songs and keep only top 5
        closest_songs.sort(key=lambda x: x[1])
        closest_songs = closest_songs[:5]

        # Add neighbors to the queue if not visited
        for neighbor in graph.adj_list[current_song]["neighbors"]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    end_time = time.time()
    time_taken_ms = (end_time - start_time) * 1000

    # Print the top 5 similar songs
    print("-" * 40)
    print("Top 5 songs similar to your listening history:")
    for i, (song, _) in enumerate(closest_songs, start=1):
        print(f"{i}. {song}")

    print(f"Time taken with Graph implementation: {time_taken_ms:.2f} ms")


def hash_table_similarity(songs_list, hash_table):
    start_time = time.time()
    song_score = 0.0  # Initialize song_score as a float
    amount_songs = 0
    for song in songs_list:
        result = hash_table.search(song)[0][0] # Get result from hash table
        song_score = song_score + result
        amount_songs += 1
    average_of_songs = song_score/amount_songs
    closest_songs = []
    for bucket in hash_table.table:
        if bucket:  # Check if the bucket is not empty
            for song, value in bucket:
                try:
                    song_metric = float(value[0])  # Assuming value[0] is the metric
                    difference = abs(song_metric - average_of_songs)
                    closest_songs.append((song, difference))
                except (TypeError, IndexError, ValueError) as e:
                    print(f"Error processing hash table song {song}: {e}")

    # Sort by the smallest difference and get the top 5
    closest_songs.sort(key=lambda x: x[1])
    top_5_closest = closest_songs[:5]

    end_time = time.time()  # End the timer
    time_taken_ms = (end_time - start_time) * 1000  # Calculate the time taken in milliseconds

    print("-" * 40)
    print("5 Songs You Might Like Based off Listening History")
    i = 1
    for song, _ in top_5_closest:
        print(f"{i}. {song}")
        i += 1

    print(f"Time taken with Hash Table implementation: {time_taken_ms:.2f} ms")  # Print the time taken
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        return
    
    file_name = sys.argv[1]
    
    graph = Graph()
    process_csv_for_graph(file_name, graph)

    hash_table = HashTable(45000)
    process_csv_for_hash_table(file_name, hash_table)

    print("Welcome to SpotiMatch")
    print("-"*40)
    songs_amount = int(input("List the amount of songs you want to put into the matcher (1 - 10): "))
    
    songs_list = []
    for i in range(songs_amount):
        song_name = input(f'Song {i+1}: ')
        if song_name in graph.adj_list or hash_table.search(song_name)[0]:
            songs_list.append(song_name)
            print("Successful")
        else:
            print(f"Song '{song_name}' not found. Please try another.")
            continue

    hash_table_similarity(songs_list, hash_table)
    graph_similarity(songs_list, graph)

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

