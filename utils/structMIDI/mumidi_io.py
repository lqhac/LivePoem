import collections
import os
import traceback
import miditoolkit
import numpy as np

STEP_PER_BAR = 32  # steps per bar
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]
DEFAULT_TICKS_PER_BAR = 480
TICKS_PER_STEP = DEFAULT_TICKS_PER_BAR * 4 // STEP_PER_BAR
STRUCT = {"Intro":1,"Verse":2, "Pre-chorus":3, "Chorus":4, "Re-intro":5, "Bridge":6,"Outro":7, "Others":8}

class Item(object):
    def __init__(self, name, start, end, vel=0, pitch=0, track=0, value='',priority = -1):
        self.name = name
        self.start = start  # start step
        self.end = end  # end step
        self.vel = vel # velocity
        self.pitch = pitch
        self.track = track  # [Drum, Piano, Guitar, Bass, Strings] if track = 0 , means this Item is belong to Chord Item
        self.value = value # Chord type or Structure type
        self.priority = priority # priority: Structure =1, Chord =2, Drum|On =3

    def __repr__(self):
        return f'Item(name={self.name:>10s}, start={self.start:>4d}, end={self.end:>4d}, ' \
               f'vel={self.vel:>3d}, pitch={self.pitch:>3d}, track={self.track:>2d}, ' \
               f'value={self.value:>10s}, priority={self.priority:>2d})\n'

    def __eq__(self, other):
        return self.name == other.name and self.start == other.start and \
               self.pitch == other.pitch and self.track == other.track and self.priority == other.priority


class Event(object):
    def __init__(self, name, value, bar, pos, track=0, dur=0, vel=0):
        self.name = name
        self.value = value
        self.bar = bar
        self.pos = pos
        self.track = track
        self.dur = dur
        self.vel = vel

    def __repr__(self):
        return f'Event(name={self.name:>10s}, value={self.value}, bar={self.bar:>2d}, ' \
               f'pos={self.pos:>2d}, track={self.track:>2d}, dur={self.dur:>2d}, ' \
               f'vel={self.vel:>2d})\n1'


def tick2step(tick, file_resolution):
    return round(tick / file_resolution * DEFAULT_TICKS_PER_BAR / TICKS_PER_STEP)


def midi2items(file_path_or_midi, instru2track, pitch_shift=0, before_infer_chords=False):
    """

    :param file_path_or_midi:
    :param instru2track:
    :param pitch_shift:
    :param before_infer_chords: 抽和弦时用的单位是tick,此值为True时,item的start、end单位为tick
    :return:
    """
    if isinstance(file_path_or_midi, str):
        midi_obj = miditoolkit.midi.parser.MidiFile(file_path_or_midi)
    else:
        midi_obj = file_path_or_midi
    file_resolution = midi_obj.ticks_per_beat  # 已统一 480

    # add note items
    note_items = []
    for instru_idx in range(len(midi_obj.instruments)):
        notes = midi_obj.instruments[instru_idx].notes
        # instrument name
        instru_name = midi_obj.instruments[instru_idx].name
        # Traci id
        instru_id = instru2track[instru_name]

        notes.sort(key=lambda x: (x.start, x.pitch))
        for note in notes:
            note_items.append(Item(
                name='On' if instru_name != 'Drums' else 'Drums',
                start=tick2step(note.start, file_resolution) if not before_infer_chords else round(note.start),
                end=tick2step(note.end, file_resolution) if not before_infer_chords else round(note.end),
                vel=note.velocity,
                pitch=note.pitch + pitch_shift,
                track=instru_id,
                priority=3

            ))

    # add chord items
    chord_items = []
    for m in midi_obj.markers:
        if m.text not in STRUCT.keys():
            chord_items.append(Item(
                name="Chord",
                start=tick2step(m.time, file_resolution),
                end=tick2step(m.time, file_resolution),
                value=m.text,
                priority=2
            ))

    # add Structure items
    struct_items = []
    for m in midi_obj.markers:
        if m.text in STRUCT.keys():
            struct_items.append(Item(
                name="Struct",
                start=tick2step(m.time, file_resolution),
                end=tick2step(m.time, file_resolution),
                value=m.text,
                priority=1
            ))

    # 合并 Note 和 Chord Item 信息，并按照时间进行排序
    items = note_items + chord_items + struct_items
    items.sort(key=lambda x: (x.start, x.priority, x.track, x.pitch, -x.end))

    # delete all Chord "N.C" items; We only consider 84 possible chord symbols
    items_ = []
    for item in items:
        if len(items_) == 0 or item.value != "N.C.":  # 开头和结尾
            items_.append(item)
    items = items_

    # 当末尾没有多余的音符信息时，去除多余的和弦token
    index_tempo = -1
    for item in reversed(items):
        print(item.priority)
        if item.priority != 3:
            index_tempo -= 1
        else:
            break
    items = items[:index_tempo + 1]

    # tempo 取平均值
    tempo = [t.tempo for t in midi_obj.tempo_changes]
    tempo = int(round(np.mean(tempo).item()))
    return items, tempo
'''
TODO: 修改到这里结束

'''

def items2events(items, n_bars):
    last_note_pos = -1
    last_instru = -1
    events = []
    bar_id = -1
    for item in items:
        # (1) Bar and Position Event
        note_bar = item.start // STEP_PER_BAR  # Bar; STEP_PER_BAR = 32，" / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。
        note_pos = item.start % STEP_PER_BAR  # Position (32; 0-31); 求模运算,相当于mod,也就是计算除法的余数,比如5%2就得到1
        while bar_id < note_bar:
            bar_id += 1
            events.append(Event(name='Bar', value=0, bar=bar_id, pos=0, track=0, dur=0, vel=0))
            last_note_pos = -1

        if note_pos != last_note_pos:
            # voc_size: [0, fraction-1]
            events.append(Event(name='Position', value=note_pos, bar=bar_id, pos=note_pos, track=0, dur=0, vel=0))
            last_note_pos = note_pos
            last_instru = -1

        # (2) Chord Event (Chord Item) |  Instrument Event and Note Event
        if item.name == 'Chord':
            events.append(Event(name='Chord', value=item.value, bar=bar_id, pos=note_pos, track=0, dur=0, vel=0))
        else:
            if item.track != last_instru:
                events.append(Event(
                    name='Instrument', value=item.track, bar=bar_id, pos=note_pos, track=item.track, dur=0, vel=0))
                last_instru = item.track
            # np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。
            velocity_index = np.clip(item.vel // 4, 1, 31).item()  # vel: [1,31], 32取不到，velocity = [0-127] //4
            duration = np.clip(item.end - item.start, 1, 64).item()  # dur: [1,63], 一个小节32个timesteps，最长不超过2小节
            events.append(Event(
                name=item.name, value=item.pitch, bar=bar_id, pos=note_pos,
                track=item.track, dur=duration, vel=velocity_index))

    # 剩余部分
    while bar_id < n_bars:
        bar_id += 1
        events.append(Event(name='Bar', value=0, bar=bar_id, pos=0, track=0, dur=0, vel=0))
    return events


def get_tempo_class(tempos_mean):
    # tempo<=90
    if tempos_mean in DEFAULT_TEMPO_INTERVALS[0] or tempos_mean <= DEFAULT_TEMPO_INTERVALS[0].start:
        tempo = 0
    # 90 < tempo <= 150
    elif tempos_mean in DEFAULT_TEMPO_INTERVALS[1]:
        tempo = 1
    # tempo > 150
    elif tempos_mean in DEFAULT_TEMPO_INTERVALS[2] or tempos_mean >= DEFAULT_TEMPO_INTERVALS[2].stop:
        tempo = 2
    return tempo


def midi_to_training_events(input_path, token2id, instru2track, cond_tracks, tgt_tracks=None, genre_tag=None, use_genre=True):
    try:
        if isinstance(input_path, str):
            item_name = os.path.basename(input_path).split(".")[0]
        else:
            input_path = item_name = ''

        # midi2item
        note_items, tempo = midi2items(input_path, instru2track=instru2track)
        # obtain tempo class
        tempo_cls = get_tempo_class(tempo)
        # get max bars
        n_bars = 0
        for n in note_items:
            note_bar = n.start // STEP_PER_BAR
            n_bars = max(n_bars, note_bar)
        # Note item sorted
        if cond_tracks is not None:
            note_items.sort(key=lambda x: (
                x.start // STEP_PER_BAR,  # bar
                -(x.track in cond_tracks),  # sorted according to the condition track
                x.start % STEP_PER_BAR,  # step in bar
                x.track, x.pitch, -x.end))
        else:
            note_items.sort(key=lambda x: (x.start, x.track, x.pitch, -x.end))

        track2instru = {v: k for k, v in instru2track.items()}
        if instru2track is not None:
            assert 0 not in list(instru2track.values()), 'Instrument ID cannot be 0. ID0 is reserved for Chord.'
        track2instru[0] = 'Chord'
        # 将 MIDI 文件中包含的轨道分到 condition_track 和 target_track并排序
        cond_tracks = sorted(set([n.track for n in note_items if track2instru[n.track] in cond_tracks]))
        # print("cond_tracks = ", cond_tracks) # e.g. cond_tracks =  [6] or cond_tracks =  [0, 6] if if condition tracks contains "Chord"
        tgt_tracks = sorted(set([n.track for n in note_items if track2instru[n.track] in tgt_tracks]))
        # print("tgt_tracks = ",tgt_tracks ) # tgt_tracks =  [6]
        if len(tgt_tracks) == 0:
            return None

        # Item2Event
        cond_events = items2events([n for n in note_items if n.track in cond_tracks], n_bars) # Chord Item in this variable if condition tracks contains "Chord"
        tgt_events = items2events([n for n in note_items if n.track in tgt_tracks], n_bars)

        # Group By Bar  按照小节进行组装
        def group_by_bar(events):
            tokens_group_by_bar = []
            for e in events:
                # Bar Event
                if e.name == 'Bar':
                    tokens_group_by_bar.append([])
                # event2dictionary
                tokens_group_by_bar[-1].append({
                    'token': token2id[f'{e.name}_{e.value}'],
                    'bar': e.bar,
                    'pos': e.pos,
                    'vel': e.vel,
                    'dur': e.dur,
                    'track': e.track,
                })
            return tokens_group_by_bar

        cond_bars = group_by_bar(cond_events)
        tgt_bars = group_by_bar(tgt_events)
        assert len(cond_bars) == len(tgt_bars), (len(cond_bars), len(tgt_bars))

        cond_length = len([t for b in cond_bars for t in b])
        tgt_length = len([t for b in tgt_bars for t in b])
        item = {
            'input_path': input_path,
            'item_name': item_name,
            'tempo': tempo_cls,
            # 'chord_items': chord_items,
            'cond_bars': cond_bars,
            'tgt_bars': tgt_bars,
            'cond_tracks': cond_tracks,
            'tgt_tracks': tgt_tracks,
            'cond_length': cond_length,
            'tgt_length': tgt_length,
        }

        # # add genre; Genre condition
        if use_genre:
            genre = input_path.split("/")[-2]
            item['genre'] = genre

        return item
    except Exception as e:
        traceback.print_exc()
        print(e, input_path)
        return None


def events2midi(track_groups, instru2program, track2instru, output_path=None,
                tempo_cls=1, tempo=None, track_velocity_limits=None, max_bars=None):
    instru2notes = collections.defaultdict(lambda: [])
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = DEFAULT_TICKS_PER_BAR
    ticks_per_bar = DEFAULT_TICKS_PER_BAR * 4  # assume 4/4

    for tracks in track_groups:
        bar = -1
        pos = -1
        track_id = -1
        for token, vel, dur in tracks:
            if token in ['<s>', '<pad>']:
                continue
            name, value = token.split("_")
            if name == 'Bar':
                bar += 1
                track_id = -1
                pos = -1
                if max_bars is not None and bar == max_bars:
                    break
            elif bar >= 0:
                if name == 'Position':
                    pos = int(value)
                    track_id = -1
                elif name == 'Instrument':
                    track_id = int(value)
                elif name in ['On', 'Drums']:
                    if track_id >= 0 and pos >= 0:
                        instru = track2instru[track_id]
                        if name == 'On' and instru == 'Drums' or name == 'Drums' and instru != 'Drums':
                            print(f"| skip token: {token}, name: {name}, track_id: {track_id}.")
                            continue
                        pitch = int(value)
                        vel = int(vel) * 4
                        if track_velocity_limits is not None and track_id in track_velocity_limits:
                            vel = np.clip(vel, track_velocity_limits[track_id][0], track_velocity_limits[track_id][1])
                        duration = int(dur) * TICKS_PER_STEP
                        start_ticks = bar * ticks_per_bar + pos * TICKS_PER_STEP
                        instru2notes[track2instru[track_id]].append(
                            miditoolkit.Note(vel, pitch, start_ticks, start_ticks + duration))
                    else:
                        print(f"| skip token: {token}, track_id: {track_id}, bar: {bar}, pos: {pos}.")
                elif name == 'Chord':
                    if track_id == -1 and pos >= 0:
                        start_ticks = bar * ticks_per_bar + pos * TICKS_PER_STEP
                        midi.markers.append(
                            miditoolkit.midi.containers.Marker(text=value, time=start_ticks))
                    else:
                        print(f"| skip token: {token}, track_id: {track_id}, bar: {bar}, pos: {pos}.")

    for k, v in instru2notes.items():
        inst = miditoolkit.midi.containers.Instrument(instru2program[k], is_drum=k == 'Drums', name=k)
        inst.notes = v
        midi.instruments.append(inst)

    # write tempo
    tempo_changes = []
    if tempo is None:
        if tempo_cls == 0:
            bpm = 60
        elif tempo_cls == 1:
            bpm = 120
        else:
            bpm = 180
    else:
        bpm = tempo
    # bpm = 140 # 恢复：把注释去掉，然后把这句注释
    tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, 0))
    midi.tempo_changes = tempo_changes
    if output_path is not None:
        midi.dump(output_path)
    return midi
