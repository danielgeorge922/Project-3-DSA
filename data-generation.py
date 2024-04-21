import csv

data = ["artist_name", "ms_played"]

artists = []
ms_played = []


with open('artist_list', 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        artists.append(row['artist_name'])

