/**
 * WebGPU Precomputed Atmospheric Scattering Demo
 *
 * This application renders a physically-based sky and atmosphere using Eric Bruneton's
 * precomputed atmospheric scattering model. The implementation uses WebGPU for rendering
 * and relies on precomputed lookup tables (LUTs) for transmittance, scattering, and irradiance.
 *
 * Key Features:
 * - Interactive camera and sun positioning
 * - Real-time atmospheric scattering with accurate Rayleigh and Mie effects
 * - Exposure control for different viewing conditions
 * - Multiple preset views showcasing various atmospheric conditions
 */

// Import utility functions for texture loading and buffer management
import {
  loadPrecomputedTextures,     // Loads transmittance, scattering, and irradiance LUTs
  createFullscreenQuadBuffer,  // Creates vertex buffer for fullscreen quad rendering
  createGlobalUniformBuffer,   // Creates uniform buffer for shader parameters
  writeGlobalUniforms,         // Writes uniform data to GPU buffer
} from './utils.js';

// Import WGSL shader code as raw string (Vite handles this via ?raw suffix)
import skyShaderWGSL from './shader.wgsl?raw';

// Get references to DOM elements
const canvas = document.getElementById('webgpu-canvas');
const helpElement = document.getElementById('help');
const statusElement = document.getElementById('status-message');
let statusMessageTimeout = null;

/**
 * Shows a centered status message and hides the instruction overlay
 *
 * @param {string} message - Message to display to the user
 */
function showStatusMessage(message) {
  if (statusElement) {
    statusElement.textContent = message;
    statusElement.classList.remove('hidden');

    // Restart the dismissal timer whenever a new status message is shown.
    if (statusMessageTimeout !== null) {
      clearTimeout(statusMessageTimeout);
    }
    statusMessageTimeout = setTimeout(() => {
      statusElement.classList.add('hidden');
      statusMessageTimeout = null;
    }, 2200);
  }
  if (helpElement) {
    helpElement.classList.add('hidden');
  }
}

/**
 * Physical Constants
 *
 * LENGTH_UNIT_IN_METERS: The model uses kilometers as the base unit (1000 meters)
 * This matches the scale of Earth's atmosphere (tens to hundreds of km)
 */
const LENGTH_UNIT_IN_METERS = 1000.0;
const PLANET_RADIUS_KM = 6360.0;
const PLANET_CENTER = [0, 0, -PLANET_RADIUS_KM];

/**
 * SUN_ANGULAR_RADIUS: The sun's angular radius as seen from Earth (in radians)
 * The sun subtends approximately 0.00935 radians (0.536 degrees) in the sky
 * We use half of this for the radius calculation
 */
const SUN_ANGULAR_RADIUS = 0.00935 / 2.0;

/**
 * DEFAULT_STATE: Initial camera and rendering parameters
 *
 * - viewDistanceMeters: Camera distance from Earth's center (9km = near ground level)
 * - viewZenithAngleRadians: Camera's zenith angle (0 = looking up, π/2 = horizon, π = looking down)
 * - viewAzimuthAngleRadians: Camera's azimuth angle (horizontal rotation)
 * - sunZenithAngleRadians: Sun's zenith angle (controls time of day)
 * - sunAzimuthAngleRadians: Sun's azimuth angle (controls sun position on horizon)
 * - exposure: Tone mapping exposure value (higher = brighter image)
 */
const DEFAULT_STATE = {
  viewDistanceMeters: 9000,
  viewZenithAngleRadians: 1.47,
  viewAzimuthAngleRadians: -0.1,
  sunZenithAngleRadians: 1.3,
  sunAzimuthAngleRadians: 2.9,
  exposure: 10,
  specularStrength: 0.35,
  orthographicCamera: false,
  freeCamera: false, freeCameraPosition: null, freeCameraYaw: 0, freeCameraPitch: 0, freeCameraRoll: 0,
};

/**
 * PRESETS: Predefined camera and sun configurations (accessible via number keys 1-9)
 *
 * Each preset showcases different atmospheric conditions:
 * - Ground-level views with various sun angles (presets 1-7)
 * - High-altitude view showing Earth's curvature (preset 8: 2700km altitude)
 * - Space view showing full atmospheric shell (preset 9: 12000km altitude)
 */
const PRESETS = {
  '1': { viewDistanceMeters: 9000, viewZenithAngleRadians: 1.47, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 1.3, sunAzimuthAngleRadians: 3.0, exposure: 10 },
  '2': { viewDistanceMeters: 9000, viewZenithAngleRadians: 1.47, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 1.564, sunAzimuthAngleRadians: -3.0, exposure: 10 },
  '3': { viewDistanceMeters: 7000, viewZenithAngleRadians: 1.57, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 1.54, sunAzimuthAngleRadians: -2.96, exposure: 10 },
  '4': { viewDistanceMeters: 7000, viewZenithAngleRadians: 1.57, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 1.328, sunAzimuthAngleRadians: -3.044, exposure: 10 },
  '5': { viewDistanceMeters: 9000, viewZenithAngleRadians: 1.39, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 1.2, sunAzimuthAngleRadians: 0.7, exposure: 10 },
  '6': { viewDistanceMeters: 9000, viewZenithAngleRadians: 1.5, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 1.628, sunAzimuthAngleRadians: 1.05, exposure: 200 },
  '7': { viewDistanceMeters: 7000, viewZenithAngleRadians: 1.43, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 1.57, sunAzimuthAngleRadians: 1.34, exposure: 40 },
  '8': { viewDistanceMeters: 2.7e6, viewZenithAngleRadians: 0.81, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 1.57, sunAzimuthAngleRadians: 2.0, exposure: 10 },
  '9': { viewDistanceMeters: 1.2e7, viewZenithAngleRadians: 0.0, viewAzimuthAngleRadians: 0, sunZenithAngleRadians: 0.93, sunAzimuthAngleRadians: -2.0, exposure: 10 },
};

/**
 * Reusable matrices and uniform buffer storage
 *
 * These are allocated once and reused each frame to avoid garbage collection:
 * - viewFromClip: Inverse projection matrix (transforms clip space to view space)
 * - modelFromView: Inverse view matrix (transforms view space to world/model space)
 * - uniformScratch: Scratch buffer for packing all uniform data before GPU upload
 */
const viewFromClip = new Float32Array(16);
const modelFromView = new Float32Array(16);
const uniformScratch = new Float32Array(64);

/**
 * Converts a 4x4 matrix from row-major to column-major layout
 *
 * JavaScript arrays are naturally row-major, but WGSL matrices are column-major.
 * This function performs the transpose during the copy operation.
 *
 * @param {Float32Array} target - Destination array
 * @param {number} offset - Starting index in target array (in float32 elements)
 * @param {Float32Array} source - Source matrix in row-major order
 */
function copyRowMajorToColumnMajor(target, offset, source) {
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      target[offset + column * 4 + row] = source[row * 4 + column];
    }
  }
}

/**
 * Initializes WebGPU device, adapter, and canvas context
 *
 * This function:
 * 1. Checks for secure context (HTTPS or localhost) - required for WebGPU
 * 2. Requests a GPU adapter with high-performance preference
 * 3. Requests a device with float32-filterable feature (for linear filtering on float textures)
 * 4. Configures the canvas context with proper DPI scaling
 *
 * @param {HTMLCanvasElement} targetCanvas - The canvas element to render to
 * @returns {Promise<{device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat}>}
 */
async function initWebGPU(targetCanvas) {
  // Check if WebGPU is available in the browser
  if (!navigator.gpu) {
    const message =
        'WebGPU is not available in this browser. Please use a WebGPU-enabled browser/version.';
    showStatusMessage(message);
    throw new Error(message);
  }

  // Request a GPU adapter (represents a physical GPU)
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('Failed to acquire GPU adapter.');
  }

  // Request a device with float32-filterable feature
  // This feature enables linear filtering on rgba32float textures (used for our LUTs)
  const device = await adapter.requestDevice({
    requiredFeatures: ['float32-filterable'],
  });

  // Get WebGPU context from canvas
  const context = targetCanvas.getContext('webgpu');
  if (!context) {
    throw new Error('Failed to acquire WebGPU context.');
  }

  // Get the preferred canvas format for this system (usually 'bgra8unorm')
  const format = navigator.gpu.getPreferredCanvasFormat();
  configureContext(targetCanvas, context, device, format);

  return { device, context, format };
}

/**
 * Configures the WebGPU canvas context with proper resolution and format
 *
 * This function handles high-DPI displays by scaling the canvas resolution
 * based on devicePixelRatio. On retina displays (devicePixelRatio = 2), this
 * doubles the resolution for crisp rendering.
 *
 * @param {HTMLCanvasElement} targetCanvas - The canvas to configure
 * @param {GPUCanvasContext} context - WebGPU canvas context
 * @param {GPUDevice} device - WebGPU device
 * @param {GPUTextureFormat} format - Texture format for the canvas
 */
function configureContext(targetCanvas, context, device, format) {
  const devicePixelRatio = window.devicePixelRatio || 1;
  targetCanvas.width = Math.max(1, Math.floor(targetCanvas.clientWidth * devicePixelRatio));
  targetCanvas.height = Math.max(1, Math.floor(targetCanvas.clientHeight * devicePixelRatio));
  context.configure({ device, format, alphaMode: 'opaque' });
}

/**
 * Creates the WebGPU render pipeline
 *
 * The pipeline defines how vertices are processed and how fragments are shaded:
 * - Vertex shader (vs_main): Transforms fullscreen quad vertices
 * - Fragment shader (fs_main): Computes atmospheric scattering for each pixel
 * - Topology: triangle-strip (4 vertices form 2 triangles = fullscreen quad)
 *
 * @param {GPUDevice} device - WebGPU device
 * @param {GPUTextureFormat} format - Output texture format
 * @returns {Promise<GPURenderPipeline>}
 */
async function initPipeline(device, format) {
  // Create shader module from WGSL code
  const shaderModule = device.createShaderModule({ code: skyShaderWGSL });

  const descriptor = {
    layout: 'auto', // Automatically infer bind group layout from shader

    // Vertex stage configuration
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [
        {
          arrayStride: 8, // 2 floats × 4 bytes = 8 bytes per vertex
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' }, // 2D position
          ],
        },
      ],
    },

    // Fragment stage configuration
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{ format }], // Output to canvas format
    },

    // Primitive assembly: triangle-strip lets us draw a quad with 4 vertices
    primitive: { topology: 'triangle-strip' },
  };

  return device.createRenderPipeline(descriptor);
}

/**
 * Renders a single frame to the canvas
 *
 * This function:
 * 1. Creates a command encoder to record GPU commands
 * 2. Begins a render pass with the canvas as the target
 * 3. Sets the pipeline, bind group (uniforms + textures), and vertex buffer
 * 4. Draws 4 vertices as a triangle-strip (forming a fullscreen quad)
 * 5. Submits the commands to the GPU queue
 *
 * @param {GPUDevice} device - WebGPU device
 * @param {GPUCanvasContext} context - Canvas context
 * @param {GPURenderPipeline} pipeline - Render pipeline
 * @param {GPUBindGroup} bindGroup - Bind group containing uniforms and textures
 * @param {GPUBuffer} vertexBuffer - Vertex buffer with fullscreen quad positions
 */
function draw(device, context, pipeline, bindGroup, vertexBuffer) {
  // Create a command encoder to record rendering commands
  const commandEncoder = device.createCommandEncoder();

  // Get the current canvas texture to render to
  const view = context.getCurrentTexture().createView();

  const renderPassDescriptor = {
    colorAttachments: [
      {
        view,
        clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }, // Dark blue clear color
        loadOp: 'clear',   // Clear the texture before rendering
        storeOp: 'store',  // Store the results to the texture
      },
    ],
  };

  // Begin render pass and record commands
  const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.setVertexBuffer(0, vertexBuffer);
  passEncoder.draw(4); // Draw 4 vertices (triangle-strip creates 2 triangles)
  passEncoder.end();

  // Submit the recorded commands to the GPU queue for execution
  device.queue.submit([commandEncoder.finish()]);
}

/**
 * Computes and packs all uniform data for the shader
 *
 * This function builds the camera transformation matrices and gathers all
 * rendering parameters into a single buffer that will be uploaded to the GPU.
 *
 * The uniform buffer layout matches the WGSL Globals struct:
 * - viewFromClip (mat4x4<f32>): Inverse projection matrix
 * - modelFromView (mat4x4<f32>): Inverse view matrix
 * - camera (vec3<f32>) + exposure (f32)
 * - sunDirection (vec3<f32>) + sunSize.x (f32)
 * - whitePoint (vec3<f32>) + sunSize.y (f32)
 * - earthCenter (vec3<f32>) + padding (f32)
 *
 * @param {HTMLCanvasElement} canvas - Canvas element (for aspect ratio)
 * @param {Object} state - Current camera and sun state
 * @returns {Float32Array} Packed uniform data ready for GPU upload
 */
function updateGlobalUniforms(canvas, state) {
  // Build inverse projection matrix (viewFromClip)
  // This transforms from clip space [-1,1] to view space
  const kFovY = (50 / 180) * Math.PI;
  const kTanFovY = Math.tan(kFovY / 2);
  const aspect = canvas.width / canvas.height;
  if (state.orthographicCamera) {
    // Orthographic rays are parallel. Scale the image to roughly match the
    // perspective view's coverage near the visible planetary horizon.
    const altitudeKm = Math.max(0.001, state.viewDistanceMeters / LENGTH_UNIT_IN_METERS);
    const horizonDistanceKm = Math.sqrt(2 * PLANET_RADIUS_KM * altitudeKm + altitudeKm * altitudeKm);
    const orthoHalfHeight = Math.max(0.01, horizonDistanceKm * kTanFovY);
    viewFromClip.set([
      orthoHalfHeight * aspect, 0, 0, 0,
      0, orthoHalfHeight, 0, 0,
      0, 0, 0, 0,
      0, 0, 1, 0,
    ]);
  } else {
    viewFromClip.set([
      kTanFovY * aspect, 0, 0, 0,
      0, kTanFovY, 0, 0,
      0, 0, 0, -1,
      0, 0, 1, 1,
    ]);
  }

  // Orbit camera or full 6DOF free camera.
  if (state.freeCamera && state.freeCameraPosition) {
    const yaw = state.freeCameraYaw, pitch = state.freeCameraPitch, roll = state.freeCameraRoll;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);
    const forward = [cp * cy, cp * sy, sp];
    const right0 = [-sy, cy, 0], up0 = [-sp * cy, -sp * sy, cp];
    const right = right0.map((v, i) => v * cr + up0[i] * sr);
    const up = up0.map((v, i) => v * cr - right0[i] * sr);
    const backward = forward.map(v => -v), pos = state.freeCameraPosition;
    modelFromView.set([right[0], up[0], backward[0], pos[0], right[1], up[1], backward[1], pos[1], right[2], up[2], backward[2], pos[2], 0, 0, 0, 1]);
  } else {
    const cosZ = Math.cos(state.viewZenithAngleRadians), sinZ = Math.sin(state.viewZenithAngleRadians);
    const cosA = Math.cos(state.viewAzimuthAngleRadians), sinA = Math.sin(state.viewAzimuthAngleRadians);
    const orbitRadius = PLANET_RADIUS_KM + state.viewDistanceMeters / LENGTH_UNIT_IN_METERS;
    const cameraOffsetX = sinZ * cosA * orbitRadius, cameraOffsetY = sinZ * sinA * orbitRadius, cameraOffsetZ = cosZ * orbitRadius;
    modelFromView.set([
      -sinA, -cosZ * cosA, sinZ * cosA, PLANET_CENTER[0] + cameraOffsetX,
      cosA, -cosZ * sinA, sinZ * sinA, PLANET_CENTER[1] + cameraOffsetY,
      0, sinZ, cosZ, PLANET_CENTER[2] + cameraOffsetZ, 0, 0, 0, 1,
    ]);
  }

  // Extract camera position from the translation column of modelFromView
  const camera = [
    modelFromView[3],
    modelFromView[7],
    modelFromView[11],
  ];

  // Compute sun direction vector from spherical coordinates
  // zenith = 0 means sun is directly overhead
  // zenith = π/2 means sun is on the horizon
  const sunDir = [
    Math.cos(state.sunAzimuthAngleRadians) * Math.sin(state.sunZenithAngleRadians),
    Math.sin(state.sunAzimuthAngleRadians) * Math.sin(state.sunZenithAngleRadians),
    Math.cos(state.sunZenithAngleRadians),
  ];

  // Sun size parameters for rendering the solar disk
  // x: tan(angular_radius) - used for disk size calculation
  // y: cos(angular_radius) - used for smooth edges
  const sunSize = [Math.tan(SUN_ANGULAR_RADIUS), Math.cos(SUN_ANGULAR_RADIUS)];

  // White point for tone mapping (neutral white)
  const whitePoint = [1, 1, 1];

  // Earth center position in world space
  // Earth radius is 6360 km, positioned below the origin
  const earthCenter = PLANET_CENTER;

  // Pack all data into the uniform buffer (converting matrices to column-major)
  copyRowMajorToColumnMajor(uniformScratch, 0, viewFromClip);   // Offset 0: mat4x4 (16 floats)
  copyRowMajorToColumnMajor(uniformScratch, 16, modelFromView); // Offset 16: mat4x4 (16 floats)
  uniformScratch.set([camera[0], camera[1], camera[2], state.exposure], 32);       // Offset 32: vec4
  uniformScratch.set([sunDir[0], sunDir[1], sunDir[2], sunSize[0]], 36);          // Offset 36: vec4
  uniformScratch.set([whitePoint[0], whitePoint[1], whitePoint[2], sunSize[1]], 40); // Offset 40: vec4
  uniformScratch.set([earthCenter[0], earthCenter[1], earthCenter[2], state.specularStrength], 44); // Offset 44: vec4 (w = specular strength)

  return uniformScratch;
}

/**
 * Applies a preset configuration to the controls
 *
 * @param {Object} controls - Control state object to update
 * @param {string} presetKey - Key for the preset (e.g., '1', '2', ..., '9')
 * @returns {boolean} True if preset was found and applied, false otherwise
 */
function applyPreset(controls, presetKey) {
  const preset = PRESETS[presetKey];
  if (!preset) {
    return false;
  }
  Object.assign(controls, preset);
  return true;
}

/**
 * Main application entry point
 *
 * This function:
 * 1. Initializes WebGPU (device, context, pipeline)
 * 2. Loads precomputed atmospheric scattering LUTs
 * 3. Creates GPU resources (buffers, bind groups)
 * 4. Sets up the render loop
 * 5. Attaches event handlers for user interaction
 */
async function loadPlanetImageTexture(device, url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  const bitmap = await createImageBitmap(await response.blob());
  const texture = device.createTexture({ label, size: [bitmap.width, bitmap.height, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
  device.queue.copyExternalImageToTexture({ source: bitmap, flipY: false }, { texture }, [bitmap.width, bitmap.height, 1]);
  bitmap.close(); return texture;
}

function createFlatHeightTexture(device, label) {
  const texture = device.createTexture({ label, size: [1, 1, 1], format: 'r16uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture }, new Uint16Array([32768]), { bytesPerRow: 2, rowsPerImage: 1 }, [1, 1, 1]);
  return texture;
}

async function loadHeightTexture(device, baseName, label) {
  // Exact 16-bit path: headerless little-endian 2048x1024 unsigned height samples.
  try {
    const response = await fetch(`./${baseName}.r16`);
    if (response.ok) {
      const bytes = await response.arrayBuffer(), width = 2048, height = 1024;
      if (bytes.byteLength !== width * height * 2) throw new Error(`${baseName}.r16 must be headerless 2048x1024 16-bit little-endian data.`);
      const texture = device.createTexture({ label: `${label} 16-bit`, size: [width, height, 1], format: 'r16uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      device.queue.writeTexture({ texture }, new Uint8Array(bytes), { bytesPerRow: width * 2, rowsPerImage: height }, [width, height, 1]); return texture;
    }
  } catch (e) { console.warn(`Unable to load ${baseName}.r16`, e); }
  // PNG fallback is 8-bit; R channel is interpreted as height.
  try {
    const response = await fetch(`./${baseName}.png`);
    if (response.ok) {
      const bitmap = await createImageBitmap(await response.blob()), cv = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = cv.getContext('2d', { willReadFrequently: true }); ctx.drawImage(bitmap, 0, 0); bitmap.close();
      const rgba = ctx.getImageData(0, 0, cv.width, cv.height).data, red = new Uint16Array(cv.width * cv.height);
      for (let i = 0; i < red.length; i++) red[i] = rgba[i * 4] * 257;
      const texture = device.createTexture({ label: `${label} 8-bit PNG`, size: [cv.width, cv.height, 1], format: 'r16uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      device.queue.writeTexture({ texture }, red, { bytesPerRow: cv.width * 2, rowsPerImage: cv.height }, [cv.width, cv.height, 1]); return texture;
    }
  } catch (e) { console.warn(`Unable to load ${baseName}.png`, e); }
  console.info(`No ${baseName} heightmap; using a flat map.`); return createFlatHeightTexture(device, `${label} flat fallback`);
}

function createProceduralCloudTexture(device) {
  const width = 512, height = 256, pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const u = x / width, v = y / height;
    const n = 0.5 + 0.20 * Math.sin(2 * Math.PI * (3 * u + 0.7 * Math.sin(2 * Math.PI * 2 * v))) + 0.15 * Math.sin(2 * Math.PI * (7 * u + 4 * v)) + 0.10 * Math.cos(2 * Math.PI * (13 * u - 9 * v)) + 0.07 * Math.sin(2 * Math.PI * (23 * u + 17 * v));
    const t = Math.max(0, Math.min(1, (n - 0.48) / 0.22)), alpha = Math.round(t * t * (3 - 2 * t) * 220), i = (y * width + x) * 4;
    pixels[i] = 246; pixels[i + 1] = 249; pixels[i + 2] = 255; pixels[i + 3] = alpha;
  }
  const texture = device.createTexture({ label: 'Procedural cloud layer fallback', size: [width, height, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture }, pixels, { bytesPerRow: width * 4, rowsPerImage: height }, [width, height, 1]); return texture;
}

async function loadCloudTexture(device) {
  try { return await loadPlanetImageTexture(device, './clouds.png', 'Cloud color and opacity map'); }
  catch { console.info('No clouds.png found; using procedural clouds.'); return createProceduralCloudTexture(device); }
}

function getOrbitCameraPosition(state) {
  const radius = PLANET_RADIUS_KM + state.viewDistanceMeters / LENGTH_UNIT_IN_METERS;
  return [PLANET_CENTER[0] + Math.sin(state.viewZenithAngleRadians) * Math.cos(state.viewAzimuthAngleRadians) * radius, PLANET_CENTER[1] + Math.sin(state.viewZenithAngleRadians) * Math.sin(state.viewAzimuthAngleRadians) * radius, PLANET_CENTER[2] + Math.cos(state.viewZenithAngleRadians) * radius];
}

async function main() {
  try {
    // Initialize WebGPU and create the render pipeline
    const gpuState = await initWebGPU(canvas);
    const pipeline = await initPipeline(gpuState.device, gpuState.format);

    // Load precomputed lookup tables (transmittance, scattering, irradiance)
    const precomputedTextures = await loadPrecomputedTextures(gpuState.device);

    // Create sampler for LUT textures
    // Linear filtering provides smooth interpolation between LUT values
    // Clamp-to-edge prevents artifacts at texture boundaries
    const lutSampler = gpuState.device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    // Planet color map and specular map use matching equirectangular UVs.
    const planetSurfaceTexture = await loadPlanetImageTexture(
      gpuState.device, './planet.png', 'Planet surface color map');
    const planetSpecularTexture = await loadPlanetImageTexture(
      gpuState.device, './planet_specular.png', 'Planet specular map');
    const cloudTexture = await loadCloudTexture(gpuState.device);
    const planetBumpTexture = await loadHeightTexture(gpuState.device, 'planet_bump', 'Planet bump map');
    const cloudBumpTexture = await loadHeightTexture(gpuState.device, 'cloud_bump', 'Cloud bump map');
    const planetSurfaceSampler = gpuState.device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
    });

    // Create vertex buffer (fullscreen quad) and uniform buffer
    const quadBuffer = createFullscreenQuadBuffer(gpuState.device);
    const uniformBuffer = createGlobalUniformBuffer(gpuState.device);

    // Create bind group that connects shader bindings to GPU resources
    const bindGroup = gpuState.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },                        // Uniforms
        { binding: 1, resource: precomputedTextures.transmittance.createView() },  // Transmittance LUT
        { binding: 2, resource: precomputedTextures.scattering.createView() },     // Scattering LUT
        { binding: 3, resource: precomputedTextures.irradiance.createView() },     // Irradiance LUT
        { binding: 4, resource: lutSampler },                                       // Sampler
        { binding: 5, resource: planetSurfaceTexture.createView() },                 // Demo sphere surface image
        { binding: 6, resource: planetSurfaceSampler },                              // Surface sampler
        { binding: 7, resource: planetSpecularTexture.createView() },                 // Planet specular map
        { binding: 8, resource: cloudTexture.createView() },                           // Cloud layer
        { binding: 9, resource: planetBumpTexture.createView() },                     // Planet bump map
        { binding: 10, resource: cloudBumpTexture.createView() },                     // Cloud bump map
      ],
    });

    // Initialize control state
    const controls = { ...DEFAULT_STATE };

    const keysDown = new Set();
    let previousFrameTime = performance.now();
    const render = (timestamp = performance.now()) => {
      const dt = Math.min(0.05, Math.max(0, (timestamp - previousFrameTime) / 1000)); previousFrameTime = timestamp;
      if (controls.freeCamera && controls.freeCameraPosition) {
        const yaw = controls.freeCameraYaw, pitch = controls.freeCameraPitch, roll = controls.freeCameraRoll;
        const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
        const forward = [cp * cy, cp * sy, sp], right0 = [-sy, cy, 0], up0 = [-sp * cy, -sp * sy, cp];
        const right = right0.map((v, i) => v * cr + up0[i] * sr), up = up0.map((v, i) => v * cr - right0[i] * sr);
        const move = [0, 0, 0], add = (basis, amount) => { for (let i = 0; i < 3; i++) move[i] += basis[i] * amount; };
        if (keysDown.has('w')) add(forward, 1); if (keysDown.has('s')) add(forward, -1);
        if (keysDown.has('d')) add(right, 1); if (keysDown.has('a')) add(right, -1);
        if (keysDown.has(' ')) add(up, 1); if (keysDown.has('control')) add(up, -1);
        const length = Math.hypot(...move);
        if (length > 0) { const speed = keysDown.has('shift') ? 2.5 : (keysDown.has('alt') ? 0.08 : 0.5); for (let i = 0; i < 3; i++) controls.freeCameraPosition[i] += move[i] / length * speed * dt; }
        if (keysDown.has('q')) controls.freeCameraRoll -= dt * 0.8; if (keysDown.has('e')) controls.freeCameraRoll += dt * 0.8;
      }
      const data = updateGlobalUniforms(canvas, controls); writeGlobalUniforms(gpuState.device, uniformBuffer, data);
      draw(gpuState.device, gpuState.context, pipeline, bindGroup, quadBuffer); requestAnimationFrame(render);
    };
    render(); // Start the render loop

    /**
     * Keyboard Controls
     *
     * +/=: Increase exposure (brighten image)
     * -: Decrease exposure (darken image)
     * [: Decrease specular strength; ]: Increase specular strength
     * o: Toggle perspective/orthographic camera mode; v: toggle 6DOF free camera
     * h: Toggle help overlay
     * 1-9: Apply preset camera/sun configurations
     */
    window.addEventListener('keydown', (event) => {
      const key = event.key.toLowerCase();
      if (['w', 'a', 's', 'd', 'q', 'e', ' ', 'control', 'shift', 'alt'].includes(key)) { keysDown.add(key); if (controls.freeCamera) event.preventDefault(); }
      if (key === 'v') {
        if (!controls.freeCamera) {
          const pos = getOrbitCameraPosition(controls), direction = [PLANET_CENTER[0] - pos[0], PLANET_CENTER[1] - pos[1], PLANET_CENTER[2] - pos[2]];
          const length = Math.hypot(...direction) || 1, dir = direction.map(v => v / length);
          controls.freeCameraPosition = pos; controls.freeCameraYaw = Math.atan2(dir[1], dir[0]);
          controls.freeCameraPitch = Math.asin(Math.max(-1, Math.min(1, dir[2]))); controls.freeCameraRoll = 0; controls.freeCamera = true;
          showStatusMessage('Freecam: WASD move, Space/Ctrl vertical, Q/E roll, drag to look');
        } else { controls.freeCamera = false; controls.freeCameraPosition = null; showStatusMessage('Orbit camera'); }
        event.preventDefault(); return;
      }
      // Exposure controls
      if (event.key === '=' || event.key === '+') {
        controls.exposure *= 1.1; // Increase by 10%
        event.preventDefault();
        return;
      }
      if (event.key === '-') {
        controls.exposure = Math.max(0.01, controls.exposure / 1.1); // Decrease by 10%, min 0.01
        event.preventDefault();
        return;
      }
      // Specular intensity controls: [ reduces, ] increases.
      if (event.key === '[') {
        controls.specularStrength = Math.max(0, controls.specularStrength - 0.05);
        event.preventDefault();
        return;
      }
      if (event.key === ']') {
        controls.specularStrength = Math.min(2, controls.specularStrength + 0.05);
        event.preventDefault();
        return;
      }

      // Toggle between perspective and orthographic projection.
      if (event.key === 'o' || event.key === 'O') {
        controls.orthographicCamera = !controls.orthographicCamera;
        showStatusMessage(`Camera: ${controls.orthographicCamera ? 'Orthographic' : 'Perspective'}`);
        event.preventDefault();
        return;
      }

      // Toggle help overlay
      if (event.key === 'h' || event.key === 'H') {
        if (helpElement) {
          helpElement.classList.toggle('hidden');
        }
        event.preventDefault();
        return;
      }

      // Preset views (1-9)
      if (applyPreset(controls, event.key)) { controls.freeCamera = false; controls.freeCameraPosition = null; event.preventDefault(); }
    });
    window.addEventListener('keyup', (event) => keysDown.delete(event.key.toLowerCase()));
    window.addEventListener('blur', () => keysDown.clear());

    /**
     * Mouse Wheel Control
     *
     * Zoom in/out by adjusting camera distance from Earth's center
     */
    canvas.addEventListener('wheel', (event) => {
      if (controls.freeCamera && controls.freeCameraPosition) {
        const yaw = controls.freeCameraYaw, pitch = controls.freeCameraPitch;
        const forward = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
        const step = event.deltaY > 0 ? -0.5 : 0.5;
        for (let i = 0; i < 3; i++) controls.freeCameraPosition[i] += forward[i] * step;
      } else { const scale = event.deltaY > 0 ? 1.05 : 1 / 1.05; controls.viewDistanceMeters = Math.max(1.0, controls.viewDistanceMeters * scale); }
      event.preventDefault();
    }, { passive: false });

    /**
     * Mouse Drag Controls
     *
     * Drag: Rotate camera (change view direction)
     * Shift+Drag: Rotate sun (change sun position)
     *
     * The drag system uses pointer events for better device compatibility
     */
    let dragMode = null;  // 'camera' or 'sun' or null
    let prevPointerX = 0;
    let prevPointerY = 0;
    const kScale = 500;   // Sensitivity factor for mouse movement

    // Start dragging
    canvas.addEventListener('pointerdown', (event) => {
      // Shift key switches to sun control mode
      dragMode = event.shiftKey ? 'sun' : 'camera';

      // Record initial pointer position (relative to canvas)
      const rect = canvas.getBoundingClientRect();
      prevPointerX = event.clientX - rect.left;
      prevPointerY = event.clientY - rect.top;

      // Capture pointer to receive move events even outside canvas
      canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    // Handle dragging
    canvas.addEventListener('pointermove', (event) => {
      if (!dragMode || !canvas.hasPointerCapture(event.pointerId)) {
        return;
      }

      // Calculate delta movement
      const rect = canvas.getBoundingClientRect();
      const pointerX = event.clientX - rect.left;
      const pointerY = event.clientY - rect.top;
      const dx = prevPointerX - pointerX;
      const dy = prevPointerY - pointerY;

      if (dragMode === 'sun') {
        // Adjust sun position
        controls.sunZenithAngleRadians -= dy / kScale;
        controls.sunZenithAngleRadians = Math.min(Math.PI, Math.max(0, controls.sunZenithAngleRadians));
        controls.sunAzimuthAngleRadians += dx / kScale;
      } else if (controls.freeCamera) {
        controls.freeCameraYaw -= dx / kScale;
        controls.freeCameraPitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, controls.freeCameraPitch + dy / kScale));
      } else {
        controls.viewZenithAngleRadians += dy / kScale;
        controls.viewZenithAngleRadians = Math.min(Math.PI - 0.01, Math.max(0.01, controls.viewZenithAngleRadians));
        controls.viewAzimuthAngleRadians += dx / kScale;
      }

      // Update previous position for next frame
      prevPointerX = pointerX;
      prevPointerY = pointerY;
    });

    // End dragging (on pointer up or cancel)
    const endDrag = (event) => {
      if (!canvas.hasPointerCapture(event.pointerId)) {
        return;
      }
      canvas.releasePointerCapture(event.pointerId);
      dragMode = null;
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    /**
     * Window Resize Handler
     *
     * Reconfigures the canvas context to match new window dimensions
     * with proper DPI scaling
     */
    window.addEventListener('resize', () => {
      configureContext(canvas, gpuState.context, gpuState.device, gpuState.format);
    });
  } catch (error) {
    console.error(error);
    const fallbackMessage =
        'Unable to start the WebGPU demo in this browser. See console for details.';
    showStatusMessage(error instanceof Error && error.message ? error.message : fallbackMessage);
  }
}

// Start the application
main();
