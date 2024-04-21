# SpotiMatch

## Introduction

SpotiMatch is a Python tool designed to analyze your music streaming history from Spotify. By processing your streaming data, SpotiMatch provides insights into your listening habits and suggests similar songs based on your preferences.

## SpotiMatch Dataset

### Description

SpotiMatch utilizes music metrics obtained from Spotify's audio features API. Each row in the dataset represents a song and includes various attributes such as danceability, energy, tempo, and genre.

### Attributes

- **danceability:** A measure of how suitable a track is for dancing, ranging from 0 to 1.
- **energy:** Represents the intensity and activity of a track, ranging from 0 to 1.
- **key:** The key of the track. Integers represent standard pitch classes (0 = C, 1 = C♯/D♭, etc.).
- **loudness:** The overall loudness of the track in decibels (dB).
- **mode:** Indicates the modality of the track (0 = minor, 1 = major).
- **speechiness:** Detects the presence of spoken words in the track, ranging from 0 to 1.
- **acousticness:** A measure of the acousticness of the track, ranging from 0 to 1.
- **instrumentalness:** Predicts whether a track contains no vocals, ranging from 0 to 1.
- **liveness:** Detects the presence of a live audience in the track, ranging from 0 to 1.
- **valence:** Represents the musical positiveness conveyed by a track, ranging from 0 to 1.
- **tempo:** The overall estimated tempo of the track in beats per minute (BPM).
- **duration_ms:** The duration of the track in milliseconds (ms).
- **time_signature:** An estimated overall time signature of a track. 
- **genre:** The genre of the track.
- **song_name:** The name of the song.
- **title:** Additional title information.

### Example

| danceability | energy | key | loudness | mode | speechiness | acousticness | instrumentalness | liveness | valence | tempo | duration_ms | time_signature | genre | song_name | title |
|--------------|--------|-----|----------|------|-------------|--------------|------------------|----------|---------|-------|-------------|----------------|-------|-----------|-------|
| 0.831        | 0.814  | 2   | -7.364   | 1    | 0.42        | 0.0598       | 0.0134           | 0.0556   | 0.389   | 156.985 | 124539      | 4              | Dark Trap | Mercury: Retrograde | - |
| 0.719        | 0.493  | 8   | -7.23    | 1    | 0.0794      | 0.401        | 0                | 0.118    | 0.124   | 115.08  | 224427      | 4              | Dark Trap | Pathology | - |
| ...          | ...    | ... | ...      | ...  | ...         | ...          | ...              | ...      | ...     | ...     | ...         | ...            | ...   | ...       | ...   |

## Installation and Setup

### Requirements

- Python 3
- Pip (Python package installer)

### Installation Steps

1. Clone or download the SpotiMatch repository to your local machine.
```sh
git clone <repository_url>
```

2. Install the required Python packages using pip.
```sh
pip install -r required.txt
```

## Analyzing Your Data

Once you have downloaded your streaming data from Spotify, you can analyze it using SpotiMatch.

### Data Processing

- SpotiMatch processes your streaming history data stored in CSV format.
- It extracts information such as artist name, song title, date and hour of playback, and stream length.

### Algorithm Implementation

SpotiMatch utilizes two main algorithms for analysis:

1. **Graph-Based Similarity:** Determines similar songs based on a graph representation of song metrics.
2. **Hash Table Similarity:** Calculates song similarities using a hash table approach.

### Running SpotiMatch

To run SpotiMatch, execute the Python script `spotimatch.py` and provide the path to your streaming history CSV file as a command-line argument.

Example:
```sh
python main.py genres_v2.csv
```

## Conclusion

By following these steps, you can gain valuable insights into your music listening habits and preferences with SpotiMatch. Enjoy exploring your Spotify streaming data!


