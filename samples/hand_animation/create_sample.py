"""Run with Blender 4.5: blender -b --factory-startup --python create_sample.py."""
from pathlib import Path
import bpy
import json
import math
import sys
from mathutils import Vector

OUT = Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(OUT / 'source/hand_model.blend'), load_ui=False, use_scripts=False)
scene = bpy.context.scene
scene.use_nodes = False
scene.render.use_compositing = False
scene.render.use_sequencer = False
scene.render.use_border = False
rig = bpy.data.objects['Rig']
meshes = [o for o in scene.objects if o.type == 'MESH']
for obj in scene.objects:
    obj.animation_data_clear()
for action in list(bpy.data.actions):
    bpy.data.actions.remove(action)
# Stored source poses require an add-on; this sample uses ordinary bone keyframes.
if 'sakura_poselib' in rig:
    del rig['sakura_poselib']
for bone in rig.pose.bones:
    bone.rotation_mode = 'XYZ'
    bone.location = (0, 0, 0)
    bone.rotation_euler = (0, 0, 0)
    bone.scale = (1, 1, 1)

# Exportable, self-contained studio materials (the original textures are external).
for obj in meshes:
    mat = bpy.data.materials.new('Sample_Skin' if obj.name == 'Hand' else 'Sample_Nails')
    color = (0.52, 0.255, 0.14, 1) if obj.name == 'Hand' else (0.74, 0.48, 0.35, 1)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.48
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.use_smooth = True

scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 120
# Endpoint 121 equals frame 1; playing frames 1..120 avoids a duplicate loop frame.
keys = [(1,0), (16,0), (49,1), (67,1), (105,0), (121,0)]
for frame, curl in keys:
    for b in rig.pose.bones:
        b.rotation_euler = (0,0,0)
    for finger in ['index', 'midd', 'ring', 'pinky']:
        for joint, degrees in [('prox',78), ('midd',98), ('dist',58)]:
            rig.pose.bones[f'{finger}_{joint}'].rotation_euler.z = math.radians(degrees)*curl
    rig.pose.bones['thumb_meta'].rotation_euler = (math.radians(22)*curl, math.radians(-12)*curl, math.radians(35)*curl)
    rig.pose.bones['thumb_prox'].rotation_euler.z = math.radians(48)*curl
    rig.pose.bones['thumb_dist'].rotation_euler.z = math.radians(38)*curl
    for b in rig.pose.bones:
        b.keyframe_insert(data_path='rotation_euler', frame=frame, group=b.name)
action = rig.animation_data.action
action.name = 'Hand_Open_Close_Loop'
action.use_fake_user = True
for layer in action.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for curve in bag.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = 'BEZIER'
                    key.handle_left_type = 'AUTO_CLAMPED'
                    key.handle_right_type = 'AUTO_CLAMPED'
scene.timeline_markers.clear()
for name, frame in [('OPEN',1),('CLOSE',16),('FIST',49),('RELEASE',67),('OPEN / LOOP',105)]:
    scene.timeline_markers.new(name, frame=frame)

# Source geometry uses approximately centimetre-sized coordinates.
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 0.01
bpy.ops.object.camera_add(location=(27,-62,27))
camera = bpy.context.object
camera.name = 'Preview_Camera'
target = Vector((-3,0,11))
camera.rotation_euler = (target-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 30
scene.camera = camera
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.display.shading.studiolight_rotate_z = 0.3
scene.display.shading.color_type = 'MATERIAL'
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
scene.display.shading.cavity_type = 'BOTH'
scene.display.shading.curvature_ridge_factor = 1.2
scene.display.shading.curvature_valley_factor = 1.0
scene.display.shading.show_specular_highlight = True
scene.display.shading.background_type = 'WORLD'
scene.world.color = (0.028,0.035,0.047)
scene.display.render_aa = '16'
scene.render.resolution_x = 720
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = False
scene.view_settings.view_transform = 'Standard'

rig['attribution'] = '3D Rigged Hand Model (c) 2026 Emma L. D. Lieker; CC BY-NC 4.0'
rig['sample_changes'] = 'New open/close animation, simplified materials, preview camera; 30 FPS, 4 seconds.'
rig['source_url'] = 'https://github.com/emmalieker/anatomical-hand-model'
for o in bpy.context.selected_objects:
    o.select_set(False)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
rig.show_in_front = False
scene.frame_set(1)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            space.region_3d.view_perspective = 'CAMERA'
            space.overlay.show_overlays = False
            space.shading.color_type = 'MATERIAL'
            space.shading.light = 'STUDIO'

# Remove unused source images/materials so no texture paths are required.
for mat in list(bpy.data.materials):
    if mat.users == 0:
        bpy.data.materials.remove(mat)
for img in list(bpy.data.images):
    if img.users == 0:
        bpy.data.images.remove(img)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'hand_open_close.blend'))

if '--preview-only' not in sys.argv:
    # glTF coordinates are metres; Blender's scene display units alone do not
    # scale export coordinates. Use an explicit root for the centimetre source.
    export_root = bpy.data.objects.new('Hand_Units_cm_to_m', None)
    scene.collection.objects.link(export_root)
    roots = [o for o in [rig] + meshes if o.parent is None]
    for o in roots:
        o.parent = export_root
    export_root.scale = (0.01,0.01,0.01)
    export_root.select_set(True)
    for o in meshes:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(OUT/'hand_open_close.glb'), export_format='GLB', use_selection=True,
        export_animations=True, export_frame_range=True, export_force_sampling=True,
        export_animation_mode='ACTIVE_ACTIONS', export_extras=True)
    for o in roots:
        o.parent = None
    bpy.data.objects.remove(export_root, do_unlink=True)

for frame, name in [(1,'preview_open.png'), (55,'preview_fist.png')]:
    scene.frame_set(frame)
    scene.render.filepath = str(OUT/name)
    bpy.ops.render.render(write_still=True)
if '--preview-only' not in sys.argv:
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.filepath = str(OUT/'hand_open_close.mp4')
    bpy.ops.render.render(animation=True)
print('SAMPLE_COMPLETE')
