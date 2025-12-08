import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print(f"SUPABASE_URL: {url}")
print(f"SUPABASE_KEY: {key[:20]}..." if key else "SUPABASE_KEY: Not Found")

if url and key:
    try:
        supabase = create_client(url, key)
        response = supabase.table("annotations").select("*").limit(1).execute()
        print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
else:
    print("❌ Missing credentials")