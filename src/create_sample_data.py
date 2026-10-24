import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Output directory and file paths (pure Path objects — no side effects on import)
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "sample_social_media_posts.csv"

# Sample social media posts for testing NexTrace
_SAMPLE_DATA = [
    {
        "post_id": "P001",
        "account_id": "A101",
        "text": "Breaking news about a major cyber security incident #CyberSecurity",
        "timestamp": "2026-10-20 09:00:00",
        "url": "https://example.com/news1"
    },
    {
        "post_id": "P002",
        "account_id": "A102",
        "text": "Breaking news about a major cyber security incident #CyberSecurity",
        "timestamp": "2026-10-20 09:01:00",
        "url": "https://example.com/news1"
    },
    {
        "post_id": "P003",
        "account_id": "A103",
        "text": "Breaking news about a major cyber security incident #CyberSecurity",
        "timestamp": "2026-10-20 09:02:00",
        "url": "https://example.com/news1"
    },
    {
        "post_id": "P004",
        "account_id": "A104",
        "text": "Remember to use strong passwords and enable two factor authentication.",
        "timestamp": "2026-10-20 10:30:00",
        "url": "https://example.com/security"
    },
    {
        "post_id": "P005",
        "account_id": "A105",
        "text": "Heavy rainfall reported in several areas. Please follow official weather updates.",
        "timestamp": "2026-10-20 11:15:00",
        "url": "https://example.com/weather"
    },
    {
        "post_id": "P006",
        "account_id": "A106",
        "text": "Breaking news about a major cyber security incident #CyberSecurity",
        "timestamp": "2026-10-20 09:03:00",
        "url": "https://example.com/news1"
    },
    {
        "post_id": "P007",
        "account_id": "A107",
        "text": "New government recruitment notification announced. Check the official website.",
        "timestamp": "2026-10-20 12:00:00",
        "url": "https://example.com/recruitment"
    },
    {
        "post_id": "P008",
        "account_id": "A108",
        "text": "Stay alert against suspicious links and online payment fraud.",
        "timestamp": "2026-10-20 12:15:00",
        "url": "https://example.com/fraud"
    },
    {
        "post_id": "P009",
        "account_id": "A109",
        "text": "Breaking news about a major cyber security incident #CyberSecurity",
        "timestamp": "2026-10-20 09:04:00",
        "url": "https://example.com/news1"
    },
    {
        "post_id": "P010",
        "account_id": "A110",
        "text": "Always verify news through reliable sources before sharing it.",
        "timestamp": "2026-10-20 13:00:00",
        "url": "https://example.com/media-literacy"
    }
]


def main():
    """Generate synthetic sample posts and write them to OUTPUT_FILE."""

    # Create the output directory only when we are about to write output.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(_SAMPLE_DATA)

    # Convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Save dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("Sample dataset created successfully!")
    print(f"Total posts: {len(df)}")
    print(f"Unique accounts: {df['account_id'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()