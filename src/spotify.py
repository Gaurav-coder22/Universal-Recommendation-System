import requests

def search_song_itunes(query):
    url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"
    data = requests.get(url).json()
    
    if data["resultCount"] == 0:
        return None
    return data["results"][0]


def recommend_from_artist(artist_name):
    url = f"https://itunes.apple.com/search?term={artist_name}&entity=song&limit=10"
    data = requests.get(url).json()

    if data["resultCount"] == 0:
        return []

    return data["results"]


def recommend(query):
    song = search_song_itunes(query)
    
    if not song:
        print("\n❌ Song not found on iTunes.")
        return

    track = song["trackName"]
    artist = song["artistName"]

    print(f"\n🎵 Found: {track} — {artist}")

    print("\n🔍 Searching similar songs...\n")
    similar = recommend_from_artist(artist)

    if not similar:
        print("❌ No recommendations found.")
        return

    print("🎧 Recommended Songs:\n")
    for item in similar:
        print("-", item["trackName"], "-", item["artistName"])


# RUN
query = input("Song name: ")
recommend(query)
