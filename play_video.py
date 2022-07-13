"""
$ pip install python-vlc
(python-vlc requires vlc  *64bit* installed.)

python-vlc issue: keep window open while playing
https://stackoverflow.com/questions/43272532/python-vlc-wont-start-the-player

T-Rex Video: https://www.youtube.com/watch?v=Ml5ABdE1HtM
"""

import vlc
from time import sleep

video_path = "video/Dino.mp4"

player = vlc.MediaPlayer(video_path)
# player.set_media(vlc.Media(video_path))  # or set media afterwards

player.toggle_fullscreen()
player.play()

sleep(1)  # wait while VLC is starting
while player.is_playing():
     sleep(0.1)
