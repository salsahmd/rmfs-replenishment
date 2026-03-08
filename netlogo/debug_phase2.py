#!/usr/bin/env python3
"""Diagnostic: just check pickle state after setup (no ticks)."""
import pickle, os, sys

# Only check existing state
if os.path.exists('netlogo.state'):
    with open('netlogo.state', 'rb') as f:
        u = pickle.load(f)
    print(f'Objects in pickle: {len(u._objects)}')
    print(f'_objects in __dict__: {"_objects" in u.__dict__}')
    obj_types = {}
    for o in u._objects:
        t = o.object_type
        obj_types[t] = obj_types.get(t, 0) + 1
    print(f'Object types: {obj_types}')
    print(f'tick_to_second: {u.tick_to_second}')
    print(f'_tick: {u._tick}')
    print(f'next_process_tick: {u.next_process_tick}')
    print(f'total_energy: {u.total_energy}')
    print(f'stop_and_go: {u.stop_and_go}')
    print(f'job_queue len: {len(u.job_queue)}')
    print(f'orders finished: {u.order_manager.finished_count}')
else:
    print('No netlogo.state found')
