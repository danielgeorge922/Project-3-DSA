import random
import csv


def generate_user_data():
    songs = {}
    with open('song-list', 'r', newline='', encoding='utf-8') as f:
        for i in range(0, 600):
            songs[i] = f.readline().strip()

    user_data = []
    user_data.append(['track_name', 'ms_played', 'genres'])

    entry = []
    for i in range(20000):
        song = random.choice(songs)

        listening_time = random.randint(1, 300000)

        entry = [song, listening_time]
        user_data.append(entry)

    return user_data


def user_data_generator(quantity):
    for i in range(quantity):
        data = generate_user_data()

        csv_file_path = "user_data" + str(i) + ".csv"
        with open(csv_file_path, mode='w', newline='') as file:
            # Create a csv.writer object
            writer = csv.writer(file)
            # Write data to the CSV file
            writer.writerows(data)
        print(f"CSV file '{csv_file_path}' created successfully.")

if __name__ == "__main__":
    user_data_generator(10)
