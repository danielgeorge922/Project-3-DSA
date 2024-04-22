import random
import csv


def generate_user_data(entries):
    songs = {}
    with open('author-artist_list', 'r', newline='', encoding='utf-8') as f:
        for i in range(0, 200):
            artist = f.readline().strip()
            song = f.readline().strip()
            songs[i] = artist + "," + song

    user_data = [['artist', 'song_name', 'duration_ms', 'danceability', 'energy']]


    for i in range(entries):
        index = random.randint(0, songs.__len__() - 1)

        entry = songs[index].split(",")
        entry.append(str(random.randint(1, 300000)))   # duration_ms
        entry.append(str(random.random()))                    # danceability
        entry.append(str(random.random()))                    # energy

        user_data.append(entry)

    return user_data


def user_data_generator(user_count, entries):
    for i in range(user_count):
        data = generate_user_data(entries)

        csv_file_path = "user_data" + str(i) + ".csv"
        with open(csv_file_path, mode='w', newline='') as file:
            # Create a csv.writer object
            writer = csv.writer(file)
            # Write data to the CSV file
            writer.writerows(data)
        print(f"CSV file '{csv_file_path}' created successfully.")


if __name__ == "__main__":
    user_data_generator(10, 15000)
