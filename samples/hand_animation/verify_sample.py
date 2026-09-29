"""Validate native animation and re-imported GLB in Blender."""
from pathlib import Path
import bpy, json, struct, math
from mathutils import Vector

out = Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(out/'hand_open_close.blend'), use_scripts=False)
scene = bpy.context.scene
rig = bpy.data.objects['Rig']
assert len(rig.data.bones) == 21
assert scene.render.fps == 30 and scene.frame_end == 120
def sample(frame):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    obj = bpy.data.objects['Hand'].evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [obj.matrix_world @ v.co for v in obj.data.vertices]
a,b,c=sample(1),sample(55),sample(121)
delta=max((x-y).length for x,y in zip(a,b))
seam=max((x-y).length for x,y in zip(a,c))
assert delta > 1 and seam < 1e-5
blob=(out/'hand_open_close.glb').read_bytes()
magic,version,length=struct.unpack_from('<4sII',blob)
assert magic==b'glTF' and version==2 and length==len(blob)
jslen,jstype=struct.unpack_from('<II',blob,12)
doc=json.loads(blob[20:20+jslen])
assert doc.get('skins') and doc.get('animations')
assert len(doc['skins'][0]['joints']) == 21
for acc in doc['accessors']:
    for value in acc.get('min',[]) + acc.get('max',[]):
        assert math.isfinite(value)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(out/'hand_open_close.glb'))
scene=bpy.context.scene
scene.render.fps=30
arms=[o for o in scene.objects if o.type=='ARMATURE']
assert len(arms)==1 and len(arms[0].data.bones)==21
mesh=next(o for o in scene.objects if o.type=='MESH' and 'Hand' in o.name)
def imported_points(frame):
    scene.frame_set(frame)
    obj=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [obj.matrix_world @ v.co for v in obj.data.vertices]
x,y,z=imported_points(1),imported_points(55),imported_points(120)
height=max(v.y for v in x)-min(v.y for v in x)
bounds=[max(v[i] for v in x)-min(v[i] for v in x) for i in range(3)]
assert 0.15 < max(bounds) < 0.35, bounds
import_delta=max((u-v).length for u,v in zip(x,y))
import_seam=max((u-v).length for u,v in zip(x,z))
assert import_delta > 0.01 and import_seam < 1e-5
report={'native_bones':21,'fps':30,'rendered_frames':120,'video_seconds':4,
    'native_pose_vertex_displacement_cm':delta,'native_loop_error_cm':seam,
    'glb_animation_count':len(doc['animations']), 'glb_open_bounds_m':bounds,
    'glb_pose_vertex_displacement_m':import_delta,'glb_loop_error_m':import_seam,
    'checks':'PASS: native deformation, loop closure, GLB skin/animation, reimport deformation and metre scale'}
(out/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
