from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json

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


if __name__ == "__main__":
    token = get_token()
    # print(token)

    artist_list = ["Drake", "2Pac"]

    result = search_for_track(token, "Uptown Funk")

    print(result["tracks"]["items"][0]["artists"][0]["name"])
    print(result["tracks"]["items"][0]["duration_ms"])


for artist in artist_list:
    result = search_for_artist(token, artist)
    print(artist)
    print(result["genres"])
    print()

"""
TODO: get a for-loop to add data in this format

[track_name, artist_name, duration_ms, [genres]]

so far, we have track name, artist name, and duration_ms
need a random generator to determine how many times each song is played and for how long

"""
# result = search_for_artist(token, "2Pac")
# print(result["genres"])