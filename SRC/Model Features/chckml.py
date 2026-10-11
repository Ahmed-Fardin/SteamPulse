import duckdb

file = "Data/Processed/steam_ml.csv"

con = duckdb.connect()

print("=" * 50)
print("STEAMPULSE ML DATA CHECK")
print("=" * 50)

rows = con.execute(f"""
    SELECT COUNT(*)
    FROM read_csv(
        '{file}',
        header = true,
        all_varchar = true
    )
""").fetchone()[0]

print(f"\nRows: {rows:,}")

print("\nInvalid HIT values:")

bad_hit = con.execute(f"""
    SELECT name, steam_appid, hit
    FROM read_csv(
        '{file}',
        header = true,
        all_varchar = true
    )
    WHERE hit NOT IN ('0', '1')
       OR hit IS NULL
""").fetchall()

for row in bad_hit:
    print(row)

print("\nInvalid split values:")

bad_split = con.execute(f"""
    SELECT name, steam_appid, split
    FROM read_csv(
        '{file}',
        header = true,
        all_varchar = true
    )
    WHERE split NOT IN ('train', 'validation', 'test')
       OR split IS NULL
""").fetchall()

for row in bad_split:
    print(row)

print("\nHIT distribution:")

hit = con.execute(f"""
    SELECT hit, COUNT(*)
    FROM read_csv(
        '{file}',
        header = true,
        all_varchar = true
    )
    GROUP BY hit
    ORDER BY hit
""").fetchall()

for row in hit:
    print(row)

print("\nSplit distribution:")

split = con.execute(f"""
    SELECT split, COUNT(*)
    FROM read_csv(
        '{file}',
        header = true,
        all_varchar = true
    )
    GROUP BY split
    ORDER BY split
""").fetchall()

for row in split:
    print(row)

print("\nDone.")