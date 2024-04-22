import csv
import sys
import time
import heapq
import numpy as np
import sys
from collections import deque
from heapq import heapify, heappush, heappop
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
    relevant_fields_hash = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                            'duration_ms', 'time_signature']
    relevant_fields_graph = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                             'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                             'duration_ms', 'time_signature']
    try:
        with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                try:
                    song_name = row['song_name']
                    if is_graph:
                        song_metrics = {float(row[key]) for key in relevant_fields_hash if key in row}

#                        key: (float(row[key]) if key != 'genre' else row[key]) for key in
#                                        relevant_fields_graph if key in row}
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


def top_10_songs(container, is_graph=False):
    heap = [(0, '')]
    heapify(heap)
    killmonger_count = 0
    start_time = time.time()

    if is_graph:
        pass
    else:
        for i in range(container.size):
            if container.table[i] is not None:
                for song, metric in container.table[i]:
                    if heap[0][0] == 'killmonger':
                        killmonger_count += 1
                        if killmonger_count == 2:
                            print('why')


                    if heap.__len__() == 10 and metric[0] < heap[0][0]:
                        continue
                    else:
                        heappush(heap, (metric[0], song))

                    if heap.__len__() > 10:
                        heappop(heap)


    end_time = time.time()

    print("Top 10 Songs:")
    for i, entry in enumerate(heap, start=1):
        print(f"{i}. {entry[1]}: {entry[0]:.4f} ms")

    time_taken_ms = (end_time - start_time) * 1000
    print(f"Time taken with BFS implementation: {time_taken_ms:.2f} ms")
    print()
"""        
        for key, value in container.adj_list.items():
            print(value.get('metrics')[0])
            if heap.__len__() == 10 and value[0] < heap[0][0]:
                continue
            else:
                heappush(heap, (value[0], key))

            if heap.__len__() > 10:
                heappop(heap)
"""

def main():
    while True:
        print("Welcome to SpotiMatch")
        print("-" * 40)
        print("1. Song Matcher")
        print("2. Top 10 Songs")
        print("3. Top 10 Genres")
        print("4. User Similarity")
        print("5. Graph Visualization")
        print("-" * 40)
        option = int(input("Please select an option from the menu above: [0 to quit] "))

        if option == 0:
            print("Goodbye!")
            break
        elif option == 1:
            file_name = input("Please enter your data file name. [0 to quit]: ")
            if file_name == "0":
                break
            elif not file_name.startswith("DATA/"):
                file_name = "DATA/" + file_name

#            if len(sys.argv) < 2:
#                print("Usage: python script.py <filename>")
#                return



            # Create instances of Graph and HashTable
            graph = Graph()
            hash_table = HashTable(45000)

            # Process the CSV file to populate graph and hash table
            process_csv(file_name, graph, is_graph=True)
            process_csv(file_name, hash_table)

#            print("Welcome to SpotiMatch")
#            print("-" * 40)
            try:
                songs_amount = int(input("List the amount of songs you want to put into the matcher (1 - 10): "))
                if not 1 <= songs_amount <= 10:
                    print("The number of songs must be between 1 and 10.")
                    return
            except ValueError:
                print("Invalid number of songs.")
                return

            songs_list = []
            i = 0  # Initialize the song counter
            while len(songs_list) < songs_amount:  # Continue until desired number of songs are found
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
            #file_name = input("Please enter your data file name. [0 to quit]: ")
            file_name = "DATA/genres_v2.csv"

            if file_name == "0":
                break
            elif not file_name.startswith("DATA/"):
                file_name = "DATA/" + file_name

            hash_table = HashTable(45000)
            graph = Graph()
            process_csv(file_name, hash_table)
            process_csv(file_name, graph, is_graph=True)

#            top_10_songs(hash_table)
            top_10_songs(graph, is_graph=True)

            continue_flag = input("Press Enter to continue...")
            if continue_flag:
                continue
            break


if __name__ == '__main__':
    main()

