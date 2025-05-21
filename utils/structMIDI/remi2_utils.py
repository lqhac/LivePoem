import collections
import os
import traceback

import miditoolkit
import numpy as np
from typing import List

from utils.mumidi.mumidi_io import  get_tempo_class
from utils.remi_utils import TICKS_PER_STEP, read_items, DEFAULT_TEMPO_INTERVALS, DEFAULT_FRACTION, Item, \
    DEFAULT_VELOCITY_BINS, DEFAULT_RESOLUTION
from utils import remi_utils
from utils.remi_utils.magenta_chord_recognition import infer_chords_for_sequence, _PITCH_CLASS_NAMES, \
    _CHORD_KIND_PITCHES, NO_CHORD


class Remi2Event(object):
    def __init__(self, name, value, bar, pos, instru=0, is_cond=0):
        self.name = name
        self.value = value
        self.bar = bar
        self.pos = pos
        self.instru = instru
        self.is_cond = is_cond

    def __repr__(self):
        return 'Event(name={:>10s}, value={}, bar={:>2d}, pos={:>2d}, instru={:>2d}, ' \
               'is_cond={:>1d})\n'.format(
            self.name, self.value, self.bar, self.pos, self.instru, self.is_cond)


def get_dict(instru2track, add_chords=True):
    id2token = []
    id2token.append('<pad>')
    id2token.append('<s>')
    id2token.append('Bar_0')
    if add_chords:
        id2token.append(f'Chord_{NO_CHORD}')
        for pc in _PITCH_CLASS_NAMES:
            for chord_kind in _CHORD_KIND_PITCHES:
                id2token.append(f'Chord_{pc}:{chord_kind}')
    for vel in range(32):
        id2token.append(f'Velocity_{vel}')
    for dur in range(64):
        id2token.append(f'Duration_{dur}')
    for instru in sorted(instru2track.values()):
        id2token.append(f'Instrument_{instru}')
    for p in range(DEFAULT_FRACTION):
        id2token.append(f'Position_{p}')
    for note_on in range(128):
        id2token.append(f'On_{note_on}')
    token2id = {k: v for v, k in enumerate(id2token)}
    return id2token, token2id


def midi2chords(midi, instru2track, token2id):
    note_items, _ = read_items(midi, 0, instru2track)
    notes_no_drum = [n for n in note_items if n.instru not in [1]]
    if len(notes_no_drum) > 0:
        _, chord_items = infer_chords_for_sequence(notes_no_drum)  # do not include drum
    else:
        chord_items = []
    for n in chord_items:
        n.start = int(round(n.start / TICKS_PER_STEP))
        n.end = int(round(n.end / TICKS_PER_STEP))
    return [[x.start, x.end, token2id[f'Chord_{x.value}']] for x in chord_items]


def item2events(notes, is_cond, noempty_bars):
    last_note_pos = -1
    last_instru = -1
    events = []
    bar_id = -1
    for n in notes:
        note_bar = n.start // DEFAULT_FRACTION
        note_pos = n.start % DEFAULT_FRACTION
        if note_bar >= len(noempty_bars):
            break

        while bar_id < note_bar:
            bar_id += 1
            events.append(Remi2Event('Bar', 0, bar_id, 0, is_cond=is_cond))
            last_note_pos = -1

        if note_pos != last_note_pos:
            # voc_size: [0, fraction-1]
            events.append(Remi2Event('Position', note_pos, bar_id, note_pos,
                                     is_cond=is_cond))
            last_note_pos = note_pos
            last_instru = -1
        if n.name == 'Chord':
            events.append(Remi2Event('Chord', n.value, bar_id, note_pos, is_cond=is_cond))
        else:
            if n.instru != last_instru:
                # voc_size: [0, max(instru2track.values())]
                events.append(Remi2Event('Instrument', n.instru, bar_id, note_pos,
                                         instru=n.instru, is_cond=is_cond))
                last_instru = n.instru

            # voc_size: [0,127]
            events.append(Remi2Event('On', n.pitch, bar_id, note_pos,
                                     instru=n.instru, is_cond=is_cond))

            # voc_size: [0,31]
            velocity_index = n.velocity // 4
            events.append(Remi2Event('Velocity', velocity_index, bar_id, note_pos,
                                     instru=n.instru, is_cond=is_cond))

            # voc_size: [0,63]
            duration = np.clip((n.end - n.start) - 1, 0, 63)
            events.append(Remi2Event('Duration', duration, bar_id, note_pos,
                                     instru=n.instru, is_cond=is_cond))

    while bar_id < len(noempty_bars):
        bar_id += 1
        events.append(Remi2Event('Bar', 0, bar_id, 0, is_cond=is_cond))

    return events


def midi2events(file_path, instru2track=None, cond_tracks=None, tgt_tracks=None):
    # [0, 1, 6]
    if isinstance(file_path, str):
        midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    else:
        midi_obj = file_path
    note_items, tempo_items = read_items(midi_obj, 0, instru2track)
    # tempo
    tempos = [x.pitch for x in tempo_items]
    if len(tempos) > 0:
        tempos_mean = int(round(np.mean(tempos)))
    else:
        tempos_mean = 120

    if instru2track is not None:
        assert 0 not in list(instru2track.values()), 'Instrument ID cannot be 0.'
    cond_tracks_ = sorted(set([n.instru for n in note_items if n.instru in cond_tracks]))
    if tgt_tracks is not None:
        tgt_tracks_ = sorted(set([n.instru for n in note_items if n.instru in tgt_tracks]))
    else:
        tgt_tracks_ = sorted(set([n.instru for n in note_items if n.instru not in cond_tracks]))
    if (len(cond_tracks) > 0 and len(cond_tracks_) == 0) or (len(tgt_tracks_) == 0 and 0 not in tgt_tracks):
        return [], [], tempos_mean, [], cond_tracks_, tgt_tracks_

    noempty_bars = list(sorted(set([round(n.start / TICKS_PER_STEP) // DEFAULT_FRACTION for n in note_items])))
    noempty_bar_id = 0
    for n in note_items:
        note_bar = round(n.start / TICKS_PER_STEP) // DEFAULT_FRACTION
        while noempty_bar_id < len(noempty_bars) and noempty_bars[noempty_bar_id] != note_bar:
            noempty_bar_id += 1
        deleted_bars = note_bar - noempty_bar_id
        n.start -= deleted_bars * TICKS_PER_STEP * DEFAULT_FRACTION
        n.end -= deleted_bars * TICKS_PER_STEP * DEFAULT_FRACTION

    chord_items = []
    for m in midi_obj.markers:
        chord_items.append(Item(
            name="Chord",
            start=round(m.time / midi_obj.ticks_per_beat * DEFAULT_RESOLUTION),
            end=round(m.time / midi_obj.ticks_per_beat * DEFAULT_RESOLUTION) + 1,
            value=m.text
        ))

    note_items = note_items + chord_items
    for n in note_items:
        n.start = int(round(n.start / TICKS_PER_STEP))
        n.end = int(round(n.end / TICKS_PER_STEP))

    if cond_tracks is not None:
        note_items.sort(key=lambda x: (
            x.start // DEFAULT_FRACTION,  # bar
            -(x.instru in cond_tracks),  # is condition track
            x.start % DEFAULT_FRACTION,  # step in bar
            x.instru, x.pitch, -x.end))
    else:
        note_items.sort(key=lambda x: (x.start, x.instru, x.pitch, -x.end))

    cond_tracks = sorted(set([n.instru for n in note_items if n.instru in cond_tracks]))
    if tgt_tracks is None:
        tgt_tracks = sorted(set([n.instru for n in note_items if n.instru not in cond_tracks]))
    else:
        tgt_tracks = sorted(set([n.instru for n in note_items if n.instru in tgt_tracks]))
    cond_events = item2events([n for n in note_items if n.instru in cond_tracks], 1, noempty_bars)
    tgt_events = item2events([n for n in note_items if n.instru in tgt_tracks], 0, noempty_bars)
    return tgt_events, cond_events, tempos_mean, chord_items, cond_tracks, tgt_tracks


def extract_events(input_path, token2id, instru2track, cond_tracks, tgt_tracks=None):
    try:
        tgt_events, cond_events, tempos_mean, chord_items, cond_tracks, tgt_tracks = \
            midi2events(input_path, instru2track, cond_tracks, tgt_tracks=tgt_tracks)
        if len(tgt_events) == 0:
            return None
        tempo = get_tempo_class(tempos_mean)

        def group_by_bar(events):
            tokens_group_by_bar = []
            for e in events:
                if e.name == 'Bar':
                    tokens_group_by_bar.append([])
                tokens_group_by_bar[-1].append({
                    'token': token2id[f'{e.name}_{e.value}'],
                    'bar_id': e.bar,
                    'pos_id': e.pos,
                    'instru_id': e.instru,
                    'is_cond': e.is_cond,
                })
            return tokens_group_by_bar

        cond_bars = group_by_bar(cond_events)
        tgt_bars = group_by_bar(tgt_events)
        # Event(name=  xx, value=    xx, bar= xx, pos=xx, instru= xx, is_cond=xx)
        item_name = os.path.basename(input_path).split(".")[0]
        assert len(cond_bars) == len(tgt_bars), (len(cond_bars), len(tgt_bars))
        chord_items = [[x.start, x.end, token2id[f'Chord_{x.value}']] for x in chord_items]
        item = {
            'input_path': input_path,
            'item_name': item_name,
            'tempo': tempo,
            'tempos_mean': tempos_mean,
            'cond_bars': cond_bars,
            'tgt_bars': tgt_bars,
            'chord_items': chord_items,
            'cond_tracks': cond_tracks,
            'tgt_tracks': tgt_tracks
        }
        return item, item_name, len(cond_events) + len(tgt_events)
    except Exception as e:
        traceback.print_exc()
        print(e, input_path)
        return None


def events2midi(tracks, output_path=None, tempo=1, tempos_mean=None,
                instru2program=None, program2instru=None,
                track_velocity_limits=None):
    instru2notes = collections.defaultdict(lambda: [])
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4  # assume 4/4

    for tokens in tracks:
        tokens = [x for x in tokens if x not in ['<s>', '<pad>']]
        bar_id = -1
        time = -1
        track_id = -1
        t_idx = 0
        while t_idx < len(tokens):
            name, value = tokens[t_idx].split("_")
            if name == 'Bar':
                bar_id += 1
                track_id = -1
                time = -1
            elif bar_id >= 0:
                if name == 'Position':
                    time = bar_id * ticks_per_bar + int(value) * TICKS_PER_STEP
                    track_id = -1
                elif name == 'Instrument':
                    track_id = int(value)
                elif name == 'On' and \
                        tokens[t_idx + 1].split('_')[0] == 'Velocity' and \
                        tokens[t_idx + 2].split('_')[0] == 'Duration' and track_id >= 0 and time >= 0:
                    pitch = int(tokens[t_idx].split('_')[1])
                    vel = int(tokens[t_idx + 1].split('_')[1]) * 4
                    if track_velocity_limits is not None:
                        vel = np.clip(vel, track_velocity_limits[track_id][0],
                                      track_velocity_limits[track_id][1])
                    duration = (int(tokens[t_idx + 2].split('_')[1]) + 1) * TICKS_PER_STEP
                    instru2notes[instru2program.get(track_id, track_id)].append(
                        miditoolkit.Note(vel, pitch, time, time + duration))
                    t_idx += 2
                elif name == 'Chord':
                    midi.markers.append(
                        miditoolkit.midi.containers.Marker(text=value, time=time))
                else:
                    print("| skip token: ", t_idx, tokens[t_idx], track_id,
                          tokens[t_idx].split('_')[0],
                          tokens[t_idx + 1].split('_')[0],
                          tokens[t_idx + 2].split('_')[0],
                          track_id,
                          time,
                          tokens[t_idx - 5:t_idx + 5])
            t_idx += 1

    for k, v in instru2notes.items():
        # write instrument
        if k < 128:
            inst = miditoolkit.midi.containers.Instrument(k, is_drum=False, name=program2instru[k])
        else:
            inst = miditoolkit.midi.containers.Instrument(0, is_drum=True, name=program2instru[k])
        inst.notes = v
        midi.instruments.append(inst)

    # write tempo
    tempo_changes = []
    if tempos_mean is None:
        if tempo == 0:
            bpm = 60
        elif tempo == 1:
            bpm = 120
        else:
            bpm = 180
    else:
        bpm = tempos_mean
    tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, 0))
    midi.tempo_changes = tempo_changes
    # write chord into marker
    # if len(temp_chords) > 0:
    #     for c in chords:
    #         midi_2.markers.append(
    #             miditoolkit.midi_2.containers.Marker(text=c[1], time=c[0]))
    # write
    if output_path is not None:
        midi.dump(output_path)
    return midi
