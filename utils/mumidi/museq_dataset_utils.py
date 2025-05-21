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
        # Obtain Note information
        for track_info in track_infos:
            note_idx = 0
            for bar_id in range(max_bars):
                n_notes = 0 # 计数器：每个小节的音符总数
                # 大条件（1）：音符数量限制
                # 小条件（2）： 音符开始的时间限制（Tick），不跨越小节
                while note_idx < len(track_info.notes) and \
                        track_info.notes[note_idx].start < bar_id * mf.ticks_per_beat * 4:
                    n_notes += 1
                    note_idx += 1
                if n_notes >= 1:
                    valid_bars[instru_idx, bar_id] = 1
    # 前后两个小节若非空也算有效的小节，当作留白
    valid_bars_pad = np.pad(valid_bars, [[0, 0], [2, 2]], mode='constant', constant_values=0) # 前后填充两个单位
    valid_bars = valid_bars | valid_bars_pad[:, :-4] | valid_bars_pad[:, 4:] # 或操作
    valid_bars = valid_bars.sum(0) == len(required_tracks) # 判断是否是空，若非空，则判定为有效小节（或者说有效Track）

    # 切割有效小节，至少8个小节以上
    seg_begin = -1
    seg_id = 0
    basename = os.path.basename(midi_path)[:-4]
    for bar_id in range(max_bars):
        seg_end = bar_id + 1
        if valid_bars[bar_id] and bar_id != max_bars - 1:
            if seg_begin == -1:
                seg_begin = bar_id
        elif seg_begin != -1:
            if seg_end - seg_begin >= 8:
                mf.tempo_changes = [miditoolkit.TempoChange(tempo, seg_begin)]
                mf.dump(f"{save_dir}/{basename}_TCOM{seg_id}.mid",
                        [seg_begin * mf.ticks_per_beat * 4, seg_end * mf.ticks_per_beat * 4])
                seg_id += 1
            seg_begin = -1


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
