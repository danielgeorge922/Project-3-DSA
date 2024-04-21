from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json
import random
import csv

"""
TODO:   cleanup this file, we have data generation but no API calls are being made,
        separate the API call functions to a separate file
        this file should only be data Generation
        
NOTES:  Since the bottleneck of the data generation are the API Calls, we can optimize by not using the API calls.
        we can do this at this stage since API calls are only needed to determine the track length to generate
        a listening time that makes sense.
        We can skip this step and generate random listening times for each song in miliseconds (0 - 5 mins),
        since we are not extremely interested in an accurate listening time per track so can ignore a track's 
        listening time to make sense all the time, to nearly all the time, which for our purpose is good enough.
"""
load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


# print(client_id, client_secret)

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token


def get_auth_header(token):
    return {"Authorization": "Bearer " + token}


def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"

    query_url = url + query

    result = get(query_url, headers=headers)

    json_result = json.loads(result.content)["artists"]["items"]

    if len(json_result) == 0:
        print("No Artists with this name exists ...")
        return None

    return json_result[0]


def search_for_track(token, track_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={track_name}&type=track&limit=1"

    query_url = url + query

    result = get(query_url, headers=headers)

    json_result = json.loads(result.content)

    return json_result


def generate_user_data():
    token = get_token()
    songs = {}
    with open('song-list', 'r', newline='', encoding='utf-8') as f:
        for i in range(0, 600):
            songs[i] = f.readline().strip()

    user_data = []
    user_data.append(['track_name', 'ms_played', 'genres'])

    entry = []
    for i in range(20000):
        song = random.choice(songs)

        # result = search_for_track(token, song)

        # artist = result["tracks"]["items"][0]["artists"][0]["name"]
        #track_length = result["tracks"]["items"][0]["duration_ms"]

        listening_time = random.randint(1, 300000)

        entry = [song, listening_time]
        user_data.append(entry)

    return user_data


"""
        track_genres = search_for_artist(token, artist)["genres"]

        if track_genres == []:
            track_genres = "Unknown"
        else:
            for i in track_genres:
                genres = ""
                genres += track_genres[i] + ","
"""


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
    token = get_token()
    # print(token)

    user_data_generator(10)

"""
    #artist_list = ["Drake", "2Pac"]

    result = search_for_track(token, "Uptown Funk")

    artist = result["tracks"]["items"][0]["artists"][0]["name"]
    track_length = result["tracks"]["items"][0]["duration_ms"]
    track_genres = search_for_artist(token, artist)["genres"]

    print(artist)
    print(track_length)
    print(track_genres)



for artist in artist_list:
    result = search_for_artist(token, artist)
    print(artist)
    print(result["genres"])
    print()
"""
"""
TODO: get a for-loop to add data in this format

[track_name, artist_name, duration_ms, [genres]]

so far, we have track name, artist name, and duration_ms
need a random generator to determine how many times each song is played and for how long

"""
# result = search_for_artist(token, "2Pac")
# print(result["genres"])