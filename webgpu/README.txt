Planet atmosphere patch: clouds, bump mapping, 16-bit heightmaps, and 6DOF freecam

Files to replace:
- main.js
- shader.wgsl

Keep these existing files beside index.html:
- planet.png                 Planet surface color map (required)
- planet_specular.png        Specular mask (required; white = stronger highlights)

Cloud layer:
- clouds.png                 Optional equirectangular RGBA cloud map. Alpha controls opacity.
                              If omitted, the app generates a procedural cloud pattern.
- cloud_bump.r16             Optional 16-bit heightmap (recommended), or cloud_bump.png (8-bit fallback).
- The cloud shell sits about 8 km above the ground. Its map opacity also drives planet/cloud-layer
  direct-sun shadowing, so opaque clouds cast shadows onto the planet surface.

Planet bump:
- planet_bump.r16             Optional 16-bit heightmap (recommended), or planet_bump.png (8-bit fallback).

16-bit heightmap format:
- Headerless little-endian unsigned 16-bit samples (0..65535).
- Exactly 2048 x 1024 pixels (4,194,304 bytes).
- Equirectangular layout matching planet.png / clouds.png.
- Higher values mean higher elevation. A flat neutral map is used if neither file is present.
- PNG fallback reads the red channel at 8-bit precision. The PNG fallback starts at 8-bit source precision; use the raw .r16 format for true 16-bit precision.

Controls:
- Drag: orbit camera; in freecam, drag to look around.
- Shift + drag: move the sun.
- Mouse wheel: zoom in orbit mode; move forward/back in freecam.
- O: toggle perspective / orthographic projection.
- V: toggle 6DOF free camera / orbit camera.
- Freecam WASD: move forward/back/strafe.
- Space / Ctrl: move along camera up/down.
- Q / E: roll left/right.
- Hold Shift while moving for faster movement; hold Alt for slower movement.
- [ / ]: decrease / increase specular strength.
- +/- and -: exposure adjustment (plus/equal brightens, minus darkens).
- H: toggle help overlay; 1-9: camera/sun presets (presets return to orbit mode).

Notes:
- Cloud shadows are a lightweight ray/shell test using the alpha channel of clouds.png; this is not a
  full volumetric cloud-shadow solution.
- The shader uses a procedural cloud fallback, and neutral bump maps if optional maps are absent.
- Tested: JavaScript syntax only. Please validate WebGPU shader compilation/rendering in your browser.
