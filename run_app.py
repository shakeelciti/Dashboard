#!/usr/bin/env python
"""
Run this instead of dishboard.py to automatically clear session data on startup
"""
import os
import shutil
from dishboard import app

# Clear old session and upload data on startup
def cleanup_on_startup():
    """Clear old uploads and temporary files"""
    uploads_folder = 'uploads'
    
    # Remove old uploads folder if it exists
    if os.path.exists(uploads_folder):
        try:
            shutil.rmtree(uploads_folder)
            print(f"✓ Cleared old uploads from {uploads_folder}")
        except Exception as e:
            print(f"! Could not clear uploads: {e}")
    
    # Recreate uploads folder
    os.makedirs(uploads_folder, exist_ok=True)
    print(f"✓ Created fresh {uploads_folder} folder")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING INTERACTIVE DASHBOARD")
    print("="*60)
    
    # Clean up before starting
    cleanup_on_startup()
    
    print("\n✓ App is ready!")
    print("→ Open your browser and go to: http://127.0.0.1:5000")
    print("→ Upload your data file to get started")
    print("\n" + "="*60 + "\n")
    
    # Run the app
    app.run(debug=True, port=5000)
