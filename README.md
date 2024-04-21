# SpotiMatch: Music Taste Compatibility Analysis Tool

## Team Name and Members

- **Team Name:** HarmonicAlgos
- **Team Members:** Adriel Barzola, Daniel George, Daniel Monzon

**Project Title**

- **Title:** SpotiMatch: Custom Algorithm for Music Taste Compatibility Analysis

## Problem

- **Description:** Existing music streaming services lack efficient mechanisms to connect users with similar musical tastes, limiting opportunities for shared musical experiences.

## Motivation

- **Why It's a Problem:** Enhancing connections through shared music tastes can foster deeper interpersonal relationships and enrich user experiences in digital music platforms. This addresses the growing demand for social and interactive features in music consumption platforms.

## Features

- **Solution Criteria:** The project aims to accurately quantify and present the similarity between users' music tastes, facilitating the discovery of new music and fostering social interactions based on musical compatibility.

## Data

- **Data Collection:** Spotify usage data is processed to extract relevant song metrics, including danceability, energy, and other attributes. These metrics are used to build a graph-based representation of song similarity.
- **Data Focus:** The focus is on collecting song metrics from Spotify usage data to populate a graph data structure. Each song is represented by its metrics, allowing for efficient similarity calculations.

## Tools

- **Programming Languages:** Python for data processing and algorithm implementation.
- **Libraries/Tools:** Standard Python libraries for data manipulation and analysis.

## Visuals

- **Interface Sketches:** User interface sketches depict the process of inputting song names and receiving recommendations based on music taste compatibility.

## Strategy

- **Algorithms/Data Structures:** The project utilizes graph-based and hash table-based data structures to efficiently calculate song similarity. Songs are represented as nodes in a graph, with edges indicating similarity based on song metrics.
- **Data Requirement:** Spotify usage data is processed to populate the graph and hash table data structures, enabling accurate similarity calculations.

## Distribution of Responsibility and Roles

- **Adriel Barzola:** Data preprocessing and graph implementation.
- **Daniel George:** Algorithm design and implementation.
- **Daniel Monzon:** Script development and project management.

**References**

- P. Lamere, "Social Tagging and Music Information Retrieval," Journal of New Music Research, vol. 37, no. 2, 2008.
- B. Whitman and S. Lawrence, "Inferring Descriptions and Similarity for Music from Community Metadata," International Computer Music Conference, 2002.
- Spotify Developer Documentation for leveraging Spotify Web API for possible real data integration and application interfacing.

