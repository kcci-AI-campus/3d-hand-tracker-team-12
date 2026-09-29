"""Bounded-size quality summaries and reproducible generation provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np

AGE_BINS = [0,25,50,75,100,150,250,500,1000,float('inf')]
FPS_BINS = [0,5,10,12,15,18,20,25,30,float('inf')]

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(8*1024*1024): digest.update(chunk)
    return digest.hexdigest()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)

def provenance(archive,cfg):
    from dataclasses import asdict
    root=Path(__file__).resolve().parent
    files=['gigahands_sim.py','gigahands_balanced.py','gigahands_batch.py','motion_archive.py','dataset_quality.py']
    return dict(format_version=2,config=asdict(cfg),archive_sha256=sha256(archive),
                code_sha256={name:sha256(root/name) for name in files},
                timing_profile_sha256=sha256(cfg.timing_profile) if cfg.timing_profile else None)

def clip_quality(data):
    mask=data['input_mask'];camera_counts=mask.any(-1).sum(1)
    joint_counts=mask.sum(1)
    valid=mask.any((-1,-2));age=(data['query_time'][:,None]-data['physical_capture_time'])*1000
    longest=0; missing_runs=[]; missing_ms=[]
    query=data["query_time"]
    durations=np.diff(query,append=query[-1]+(query[-1]-query[-2] if len(query)>1 else 0))*1000
    for camera in range(3):
        for hand in range(2):
            run=0
            duration=0.
            for index, missing in enumerate(~mask[:,camera,hand].any(-1)):
                if missing:
                    run+=1; duration+=durations[index]
                else:
                    if run: missing_runs.append(run); missing_ms.append(duration)
                    run=0; duration=0.
                longest=max(longest,run)
            if run: missing_runs.append(run); missing_ms.append(duration)
    events=data['hand_timing_events'];intervals=[];inference=[]
    for camera in range(3):
        if len(events):
            rows=events[events[:,0]==camera];intervals.extend(np.diff(rows[:,1])*1000);inference.extend(rows[:,4])
    meta=json.loads(str(data['metadata']))
    return dict(false_positive_hand_observations=int(data['false_positive_mask'].sum()) if 'false_positive_mask' in data else 0,
                frames=len(mask),windows=len(data['window_starts']),
                hand_camera_count=np.bincount(camera_counts.ravel(),minlength=4).tolist(),
                joint_camera_count=np.bincount(joint_counts.ravel(),minlength=4).tolist(),
                observation_age_hist=np.histogram(age[valid],AGE_BINS)[0].tolist(),
                capture_interval_hist=np.histogram(intervals,AGE_BINS)[0].tolist(),
                inference_time_hist=np.histogram(inference,AGE_BINS)[0].tolist(),
                processing_fps_hist=np.histogram(1000/np.asarray(intervals),FPS_BINS)[0].tolist(),
                missing_duration_hist=np.histogram(missing_ms,AGE_BINS)[0].tolist(),
                longest_missing_query_frames=longest,
                longest_missing_ms=max(missing_ms,default=0.),
                excluded_windows=meta['workspace']['excluded_windows'])

def dataset_report(rows,failures):
    result={'files':len(rows),'failed_clips':len(failures),'histogram_edges_ms':AGE_BINS[:-1]+['infinity'],
            'fps_edges':FPS_BINS[:-1]+['infinity'], 'scope':'all query frames, including frames excluded from training windows', 'splits':{}}
    for split in ('train','val','test'):
        selected=[r for r in rows if r['split']==split]
        q=[r['quality'] for r in selected]
        result['splits'][split]=dict(clips=len(selected),participants=len({r['participant'] for r in selected}),
            false_positive_hand_observations=sum(x.get('false_positive_hand_observations',0) for x in q),
            frames=sum(x['frames'] for x in q),windows=sum(x['windows'] for x in q),
            excluded_windows=sum(x['excluded_windows'] for x in q),
            longest_missing_ms=max((x['longest_missing_ms'] for x in q),default=0))
        for name in ('hand_camera_count','joint_camera_count','observation_age_hist','capture_interval_hist','inference_time_hist','processing_fps_hist','missing_duration_hist'):
            result['splits'][split][name]=np.sum([x[name] for x in q],axis=0).tolist() if q else []
    return result

def write_report(output,rows,failures):
    output=Path(output);report=dataset_report(rows,failures)
    atomic_json(output/'quality_report.json',report)
    lines=['# Dataset quality','',f"Files: {len(rows)} · Failed clips: {len(failures)}",'',
           '| Split | Participants | Clips | Windows | Excluded windows | Longest missing (ms) |',
           '|---|---:|---:|---:|---:|---:|']
    observations=[]
    for split,q in report['splits'].items():
        lines.append(f"| {split} | {q['participants']} | {q['clips']} | {q['windows']} | {q['excluded_windows']} | {q['longest_missing_ms']:.1f} |")
        counts=q['hand_camera_count'];total=sum(counts)
        if total:
            observations.append(split+' hand observation counts (0/1/2/3 cameras): '+', '.join(f'{100*n/total:.1f}%' for n in counts))
    lines+=['',*observations]
    lines+=['','Histograms for age, missing duration, inference time and processing FPS are in quality_report.json.',
            'Observation summaries cover all query frames; window counts exclude out-of-workspace windows.',
            'Timing histograms use hand-dependent timing events and are empty if that mode is disabled.']
    (output/'quality_report.md').write_text('\n'.join(lines),encoding='utf-8')
