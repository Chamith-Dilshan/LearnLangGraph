# pip install sqlite3 is built-in — just run: python create_playlist_db.py
import sqlite3

con = sqlite3.connect("playlist.db")
cur = con.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    genre TEXT NOT NULL,   -- 'deep house', 'jazz', 'top 40', 'reggaeton', ...
    bpm INTEGER NOT NULL,  -- ~60-90 chill/dinner, 100-120 groove, 120-140 dancefloor
    vibe TEXT NOT NULL,    -- 'chill' | 'romantic' | 'energetic' | 'party'
    energy REAL NOT NULL,  -- 0.0 ambient ... 1.0 peak-time
    year INTEGER
)
""")

songs = [
    ("Ocean Drive", "Lana Rey", "deep house", 118, "energetic", 0.72, 2019),
    ("Golden Hour", "The Midnights", "indie pop", 105, "romantic", 0.45, 2021),
    ("Velvet Sky", "Nu Era Trio", "jazz", 92, "chill", 0.25, 2018),
    ("Baila Ahora", "DJ Sol", "reggaeton", 128, "party", 0.88, 2022),
    ("Neon Hearts", "Pulse Theory", "synthwave", 112, "energetic", 0.65, 2020),
    ("Slow Bloom", "Amber Fields", "jazz", 74, "chill", 0.18, 2017),
    ("First Dance", "Elin Marsh", "soul", 84, "romantic", 0.35, 2016),
    ("Rooftop Rush", "Kite & Co.", "top 40", 124, "party", 0.85, 2023),
    ("Dusk Parade", "Mono Club", "indie pop", 100, "energetic", 0.55, 2021),
    ("Afterglow", "Selene Waves", "deep house", 122, "party", 0.80, 2022),
    ("Paper Lanterns", "Hiro Sato", "ambient", 62, "chill", 0.10, 2019),
    ("Carnival '99", "Brasa Sound", "latin jazz", 108, "energetic", 0.60, 2015),
    ("Strobe Lights", "NovaKid", "electro", 134, "party", 0.95, 2020),
    ("Honey Drip", "Mika Blue", "r&b", 88, "romantic", 0.30, 2018),
    ("Weekend Fuel", "The Statics", "top 40", 126, "party", 0.83, 2023),
]

cur.executemany(
    "INSERT INTO songs (title, artist, genre, bpm, vibe, energy, year) VALUES (?,?,?,?,?,?,?)",
    songs,
)
con.commit()
con.close()
print(f"Seeded playlist.db with {len(songs)} songs")
