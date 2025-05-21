import glob
import os
from multiprocessing.pool import Pool
import miditoolkit
import numpy as np
from tqdm import tqdm


def filter_complete_tracks(midi_path, step_per_bar, required_tracks, save_dir):
    """
    筛选并保存包含完整轨道（track_infos）的midi
    :param midi_path:
    :param step_per_bar:
    :param required_programs:
    :param save_dir:
    :return:
    """
    mf = miditoolkit.MidiFile(midi_path)
    # tempo
    tempo = mf.tempo_changes[0].tempo
    # calculate the max_bars | 1 bar(小节) = 4 beat (拍) = 32 timesteps (时间步) = 8 timesteps per beat  = 480 Ticks per beat = 480*4 Ticks
    tick_per_step = mf.ticks_per_beat * 4 / step_per_bar # mf.ticks_per_beat 已统一为480， tick_per_step = 480 * 4 /32 = 60； 即每一个 timestep = 60 ticks
    max_steps = int(mf.max_tick // tick_per_step) + 1 # 时间步总数
    max_bars = max_steps // step_per_bar + 1 # 小节总数
    valid_bars = np.zeros([len(required_tracks), max_bars], dtype=np.bool) # 统计有效的小节： 即包含音符的小节

    # for example: required_tracks = ["Lead", "Piano|Guitar","Drum"]
    for instru_idx, tracks in enumerate(required_tracks):
        # Obtain One Required Track information
        track_infos = [x for x in mf.instruments if x.name in tracks.split("|")]
        # print(track_infos) # [Instrument(program=80, is_drum=False, name="Lead")]
        if len(track_infos) == 0:
            return

    basename = os.path.basename(midi_path)[:-4]
    mf.dump(f"{save_dir}/{basename}_LSC2Others.mid")



def mp_filter_complete_tracks(data_base_dir, step_per_bar, required_tracks, save_dir):
    # create the folder for saving the filtered midi_2
    os.makedirs(save_dir, exist_ok=True)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(filter_complete_tracks, args=[path, step_per_bar, required_tracks, save_dir])
               for path in glob.glob(f'{data_base_dir}/*.mid')]
    pool.close()
    results = [x.get() for x in tqdm(list(futures))]
    pool.join()
    print(f"| Finish {len(results)} jobs.")