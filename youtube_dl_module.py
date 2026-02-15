import os
import re
import json
import yt_dlp
from mutagen.id3 import ID3, APIC, TPE1, TALB, TIT2, TCON, TDRC, COMM

class YouTubeMP3Downloader:
    """Modular YouTube MP3 Downloader using yt-dlp for everything."""
    
    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_video_info(self, url):
        """Fetch video metadata using yt-dlp (no download)."""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'noplaylist': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'author': info.get('uploader'),
                    'thumbnail': info.get('thumbnail'),
                    'duration': info.get('duration')
                }
        except Exception as e:
            print(f"Info fetch error: {e}")
            return None

    def download_thumbnail_data(self, url):
        """Download thumbnail bytes for embedding."""
        try:
            # yt-dlp doesn't have a simple 'download bytes' method for thumbs,
            # so we use requests here but with better error handling/headers
            import requests
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            return resp.content if resp.status_code == 200 else None
        except:
            return None

    def download_audio(self, url, filename):
        try:
            # Ensure filename is safe
            safe_filename = "".join([c for c in filename if c.isalnum() or c in (' ', '-', '_')]).strip()
            safe_filename = re.sub(r'\s+', ' ', safe_filename)
            if not safe_filename.endswith('.mp3'):
                safe_filename += '.mp3'
            
            output_path = os.path.join(self.output_dir, safe_filename)
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_path.replace('.mp3', ''),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the file (handling extension variations)
            if os.path.exists(output_path):
                return output_path
            
            return None
            
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    def embed_metadata(self, mp3_path, title, artist, album, thumbnail_data, url):
        try:
            audio = ID3()
            try: audio.delete(mp3_path)
            except: pass
            
            audio = ID3()
            audio.add(TIT2(encoding=3, text=title))
            audio.add(TPE1(encoding=3, text=artist))
            audio.add(TALB(encoding=3, text=album))
            audio.add(TCON(encoding=3, text='YouTube'))
            audio.add(COMM(encoding=3, lang='eng', desc='Source', text=url))
            if thumbnail_data:
                audio.add(APIC(encoding=0, mime='image/jpeg', type=3, desc='', data=thumbnail_data))
            audio.save(mp3_path, v2_version=3)
            return True
        except Exception as e:
            print(f"Metadata error: {e}")
            return False

    def process(self, url):
        info = self.get_video_info(url)
        if not info: return "Could not get video info"
        
        title = info.get('title', 'Unknown')
        artist = info.get('author', 'Unknown')
        thumb_url = info.get('thumbnail')
        
        print(f"Downloading: {title}")
        safe_name = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).strip()[:50]
        actual_file_path = self.download_audio(url, f"{safe_name}.mp3")
        
        if not actual_file_path:
            return "Download failed"
        
        print(f"File created: {actual_file_path}")
        
        thumb_data = self.download_thumbnail_data(thumb_url) if thumb_url else None
        if self.embed_metadata(actual_file_path, title, artist, artist, thumb_data, url):
            return f"Success: {os.path.basename(actual_file_path)}"
        return "Metadata embedding failed"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        downloader = YouTubeMP3Downloader()
        print(downloader.process(sys.argv[1]))
    else:
        print("Usage: python youtube_dl_module.py <youtube_url>")
