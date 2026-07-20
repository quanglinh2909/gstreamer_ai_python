import os
import vlc
import queue
import time

os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['AUDIODEV'] = 'null'


class PlaySound:
    def __init__(self):
        self.q_play_sound = queue.Queue()
        self.tim_pre = 0

    def play_sound(self):
        while True:
            try:
                data = self.q_play_sound.get()
                time_start = data.get("time", 0)
                link_mp3 = data.get("link")
                if not link_mp3 or not os.path.exists(link_mp3):
                    print(f"⚠️ Không tìm thấy file âm thanh: {link_mp3}")
                    self.q_play_sound.task_done()
                    continue

                # Kiểm tra: nếu chưa đủ 1 giây kể từ lần phát trước, bỏ qua
                time_since_last = time_start - self.tim_pre
                if time_since_last < 1.0:
                    print(f"⏭️ Bỏ qua âm thanh (chỉ {time_since_last:.2f}s từ lần trước)")
                    self.q_play_sound.task_done()
                    continue

                # Cập nhật thời điểm phát mới nhất
                self.tim_pre = time_start

                # Tạo VLC instance với options để tránh lỗi PulseAudio
                # Sử dụng dummy audio output hoặc alsa thay vì PulseAudio
                instance = vlc.Instance('--no-video', '--aout=alsa', '--quiet')
                player = instance.media_player_new()
                media = instance.media_new(link_mp3)
                player.set_media(media)
                player.play()
                print(f"▶️ Phát âm thanh: {link_mp3} tại thời điểm {time_start}")
                # đợi phát xong
                while player.get_state() != vlc.State.Ended:
                    time.sleep(0.1)
                self.q_play_sound.task_done()
            except Exception as e:
                print(f"Lỗi phát âm thanh: {e}")
                return None
            

play_sound = PlaySound()
