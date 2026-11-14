use crate::gpu::solver::GpuParticle;
use crate::math::Aabb;
use crate::render::{
    Camera, CliRenderQuality, ColorMap, ColorMapType, ColorMode, EnvironmentPreset, MeshVertex,
    SurfaceMesh, SurfaceMeshConfig,
};
use crate::sph::chamber::Chamber;
use crate::sph::collider::{Collider, ColliderShape};
use crate::sph::particle::{Particle, ParticleType};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use rayon::prelude::*;
use std::sync::Arc;

const MAX_SSAA_INTERNAL_PIXELS: u64 = 36_000_000;
const MAX_ANALYTIC_COLLIDERS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuColliderItem {
    pub min_or_center: [f32; 4],
    pub max_or_ext: [f32; 4],
    pub kind_pad: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRenderUniforms {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub shadow_vp: [[f32; 4]; 4],

    pub eye_pos: [f32; 4],
    pub cam_fwd: [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],

    pub light_dir: [f32; 4],
    pub eye_light_dir: [f32; 4],
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],

    pub bg_low: [f32; 4],
    pub bg_high: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_ground: [f32; 4],

    pub shallow_lin: [f32; 4],
    pub deep_lin: [f32; 4],
    pub shallow_srgb: [f32; 4],
    pub deep_srgb: [f32; 4],
    pub mat_opt1: [f32; 4],
    pub mat_opt2: [f32; 4],
    pub wire_color: [f32; 4],

    pub dims: [u32; 4],
    pub flags0: [u32; 4],
    pub flags1: [u32; 4],
    pub flags2: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObstacleVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
}

#[inline]
fn srgb_slice_to_linear(c: [f32; 3]) -> Vec3 {
    Vec3::new(c[0].powf(2.2), c[1].powf(2.2), c[2].powf(2.2))
}

#[inline]
fn color_mode_id(mode: ColorMode) -> u32 {
    match mode {
        ColorMode::Velocity => 0,
        ColorMode::Pressure => 1,
        ColorMode::Density => 2,
        ColorMode::Height => 3,
        ColorMode::KineticEnergy => 4,
        ColorMode::Foam => 5,
        ColorMode::Vorticity => 6,
        ColorMode::SingleColor => 7,
    }
}

#[inline]
fn color_map_id(cmap: ColorMapType) -> u32 {
    match cmap {
        ColorMapType::Ocean => 0,
        ColorMapType::Turbo => 1,
        ColorMapType::Viridis => 2,
        ColorMapType::Fire => 3,
        ColorMapType::Plasma => 4,
        ColorMapType::Inferno => 5,
        ColorMapType::Ice => 6,
        ColorMapType::Chrome => 7,
    }
}

#[inline]
fn env_preset_id(preset: EnvironmentPreset) -> u32 {
    match preset {
        EnvironmentPreset::StudioDark => 0,
        EnvironmentPreset::DaylightSky => 1,
        EnvironmentPreset::GoldenSunset => 2,
        EnvironmentPreset::ArcticLab => 3,
        EnvironmentPreset::AbyssalDeep => 4,
    }
}

fn build_color_lut_data() -> Vec<[f32; 4]> {
    let cmaps = [
        ColorMapType::Ocean,
        ColorMapType::Turbo,
        ColorMapType::Viridis,
        ColorMapType::Fire,
        ColorMapType::Plasma,
        ColorMapType::Inferno,
        ColorMapType::Ice,
        ColorMapType::Chrome,
    ];
    let mut lut = Vec::with_capacity(8 * 256);
    for cmap in cmaps {
        for i in 0..256 {
            let t = i as f32 / 255.0;
            let [r, g, b] = ColorMap::sample(cmap, t);
            lut.push([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]);
        }
    }
    lut
}

/// Hardware-accelerated headless `wgpu` 3D renderer for CLI PNG/MP4 studio output.
///
/// Mirrors `SoftwareRenderer` optics (supersampled linear HDR buffer, volumetric fluid
/// shadows & floor caustics, screen-space chromatic refraction, Beer-Lambert absorption,
/// GGX specular, studio environment reflection, whitewater foam, depth-aware chamber
/// wireframes, and direct zero-readback GPU particle sphere rendering).
pub struct GpuRenderer {
    pub width: usize,
    pub height: usize,
    pub rgb_buffer: Vec<u8>,
    pub render_quality: CliRenderQuality,
    pub ssaa_factor: usize,
    pub show_wireframe: bool,
    pub show_grid: bool,
    pub env_preset: EnvironmentPreset,
    pub exposure: f32,
    pub particle_foam: bool,
    pub lit_spheres: bool,
    pub light_dir: Vec3,
    pub light_intensity: f32,
    pub ambient_intensity: f32,
    pub device_name: String,

    render_width: usize,
    render_height: usize,
    shadow_res: usize,

    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Offscreen textures & views
    hdr_tex: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    bg_tex: wgpu::Texture,
    thickness_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    shadow_view: wgpu::TextureView,

    // Pipelines
    bg_pipeline: wgpu::RenderPipeline,
    obstacle_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_blur_h_pipeline: wgpu::ComputePipeline,
    shadow_blur_v_pipeline: wgpu::ComputePipeline,
    thickness_pipeline: wgpu::RenderPipeline,
    fluid_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::ComputePipeline,

    // Global buffers & bind groups
    uniform_buf: wgpu::Buffer,
    collider_buf: wgpu::Buffer,
    global_bg: wgpu::BindGroup,

    bg_tex_bg: wgpu::BindGroup,
    shadow_blur_h_bg: wgpu::BindGroup,
    shadow_blur_v_bg: wgpu::BindGroup,
    fluid_tex_bg: wgpu::BindGroup,
    wire_depth_bg: wgpu::BindGroup,

    particle_bgl: wgpu::BindGroupLayout,
    color_lut_buf: wgpu::Buffer,
    particle_upload_buf: Option<wgpu::Buffer>,
    particle_upload_cap: usize,
    particle_upload_bg: Option<wgpu::BindGroup>,
    cached_ext_particle_buf: Option<wgpu::Buffer>,
    cached_ext_particle_bg: Option<wgpu::BindGroup>,

    out_rgba_buf: wgpu::Buffer,
    out_read_buf: wgpu::Buffer,
    resolve_bg: wgpu::BindGroup,

    // Dynamic geometry buffers (reused across frames)
    mesh_vbo: Option<wgpu::Buffer>,
    mesh_vbo_cap: usize,
    mesh_ibo: Option<wgpu::Buffer>,
    mesh_ibo_cap: usize,
    obstacle_vbo: Option<wgpu::Buffer>,
    obstacle_vbo_cap: usize,
    wire_vbo: Option<wgpu::Buffer>,
    wire_vbo_cap: usize,

    // Reusable CPU scratch
    collider_scratch: Vec<GpuColliderItem>,
    particle_scratch: Vec<GpuParticle>,
    obstacle_scratch: Vec<ObstacleVertex>,
    wire_scratch: Vec<LineVertex>,
}

impl GpuRenderer {
    pub fn new_standalone(
        width: usize,
        height: usize,
        quality: CliRenderQuality,
    ) -> Result<Self, String> {
        let (device, queue, device_name) = crate::gpu::create_headless_wgpu_device()?;
        Self::new_with_device(device, queue, device_name, width, height, quality)
    }

    pub fn new_with_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        device_name: String,
        width: usize,
        height: usize,
        quality: CliRenderQuality,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("Output dimensions must be non-zero".to_string());
        }
        if device.limits().max_storage_buffers_per_shader_stage < 2 {
            return Err("Adapter lacks storage buffer support for GPU rendering".to_string());
        }

        let max_tex = device.limits().max_texture_dimension_2d as usize;
        if width > max_tex || height > max_tex {
            return Err(format!(
                "Requested {}x{} exceeds GPU max_texture_dimension_2d ({})",
                width, height, max_tex
            ));
        }

        let mut ssaa = quality.ssaa_factor().max(1);
        let base_px = (width as u64).saturating_mul(height as u64);
        while ssaa > 1
            && (base_px.saturating_mul((ssaa * ssaa) as u64) > MAX_SSAA_INTERNAL_PIXELS
                || width * ssaa > max_tex
                || height * ssaa > max_tex)
        {
            ssaa -= 1;
        }
        let render_width = width * ssaa;
        let render_height = height * ssaa;
        let shadow_res = quality.shadow_res().min(max_tex).max(64);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LiquidRust Render WGSL"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });

        let create_rgba16f = |label: &str, w: u32, h: u32, extra: wgpu::TextureUsages| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | extra,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };

        let (hdr_tex, hdr_view) = create_rgba16f(
            "GPU Render HDR Target",
            render_width as u32,
            render_height as u32,
            wgpu::TextureUsages::COPY_SRC,
        );
        let (bg_tex, bg_view) = create_rgba16f(
            "GPU Render BG Snapshot",
            render_width as u32,
            render_height as u32,
            wgpu::TextureUsages::COPY_DST,
        );
        let (_thickness_tex, thickness_view) = create_rgba16f(
            "GPU Render Fluid Thickness",
            render_width as u32,
            render_height as u32,
            wgpu::TextureUsages::empty(),
        );
        let (_shadow_tex, shadow_view) = create_rgba16f(
            "GPU Render Fluid Shadow Map",
            shadow_res as u32,
            shadow_res as u32,
            wgpu::TextureUsages::STORAGE_BINDING,
        );
        let (_shadow_scratch_tex, shadow_scratch_view) = create_rgba16f(
            "GPU Render Fluid Shadow Scratch",
            shadow_res as u32,
            shadow_res as u32,
            wgpu::TextureUsages::STORAGE_BINDING,
        );

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GPU Render Depth"),
            size: wgpu::Extent3d {
                width: render_width as u32,
                height: render_height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Bind Group Layouts
        let global_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Global BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bg_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BG Shadow Tex BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let fluid_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Mesh Tex BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let wire_depth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Wireframe Depth BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let particle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Storage BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Empty Group1 BGL"),
            entries: &[],
        });

        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Resolve Group2 BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shadow_blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Blur Group1 BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let add_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let make_depth_stencil = |write: bool, compare: wgpu::CompareFunction| wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: write,
            depth_compare: compare,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // 1A. Background pipeline
        let bg_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BG Pipeline Layout"),
            bind_group_layouts: &[&global_bgl, &bg_tex_bgl],
            push_constant_ranges: &[],
        });
        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("BG Render Pipeline"),
            layout: Some(&bg_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("bg_vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("bg_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(make_depth_stencil(true, wgpu::CompareFunction::Always)),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 1B. Obstacle mesh pipeline
        let global_only_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Global Only Pipeline Layout"),
            bind_group_layouts: &[&global_bgl],
            push_constant_ranges: &[],
        });
        let obstacle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Obstacle Mesh Pipeline"),
            layout: Some(&global_only_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("obstacle_vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ObstacleVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("obstacle_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(make_depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mesh_pos_only_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        };

        // 2. Shadow map pipeline
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&global_only_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("shadow_vs"),
                buffers: &[mesh_pos_only_vbl.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("shadow_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(add_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 3. Thickness pipeline
        let thickness_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Thickness Pipeline"),
            layout: Some(&global_only_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("thickness_vs"),
                buffers: &[mesh_pos_only_vbl],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("thickness_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(add_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(make_depth_stencil(false, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 4A. Fluid surface mesh pipeline
        let fluid_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Mesh Pipeline Layout"),
            bind_group_layouts: &[&global_bgl, &fluid_tex_bgl],
            push_constant_ranges: &[],
        });
        let fluid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fluid Mesh Pipeline"),
            layout: Some(&fluid_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("fluid_vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 28,
                            shader_location: 3,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fluid_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(make_depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 4B. Particle spheres pipeline
        let particle_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Pipeline Layout"),
            bind_group_layouts: &[&global_bgl, &particle_bgl],
            push_constant_ranges: &[],
        });
        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline"),
            layout: Some(&particle_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("particle_vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("particle_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(make_depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 5. Wireframe overlay pipeline
        let wire_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Wireframe Pipeline Layout"),
            bind_group_layouts: &[&global_bgl, &wire_depth_bgl],
            push_constant_ranges: &[],
        });
        let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wireframe Overlay Pipeline"),
            layout: Some(&wire_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("wireframe_vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("wireframe_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 6. Compute SSAA resolve & separable shadow blur pipelines
        let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resolve Pipeline Layout"),
            bind_group_layouts: &[&global_bgl, &empty_bgl, &resolve_bgl],
            push_constant_ranges: &[],
        });
        let resolve_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSAA Resolve Pipeline"),
            layout: Some(&resolve_pl),
            module: &shader,
            entry_point: Some("resolve_ssaa"),
            compilation_options: Default::default(),
            cache: None,
        });

        let shadow_blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Blur Pipeline Layout"),
            bind_group_layouts: &[&global_bgl, &shadow_blur_bgl],
            push_constant_ranges: &[],
        });
        let shadow_blur_h_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Shadow Blur Horizontal Pipeline"),
                layout: Some(&shadow_blur_pl),
                module: &shader,
                entry_point: Some("shadow_blur_h"),
                compilation_options: Default::default(),
                cache: None,
            });
        let shadow_blur_v_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Shadow Blur Vertical Pipeline"),
                layout: Some(&shadow_blur_pl),
                module: &shader,
                entry_point: Some("shadow_blur_v"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Global uniform & collider buffers
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Uniforms"),
            size: std::mem::size_of::<GpuRenderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let collider_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Colliders"),
            size: (MAX_ANALYTIC_COLLIDERS * std::mem::size_of::<GpuColliderItem>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let global_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Global BG"),
            layout: &global_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: collider_buf.as_entire_binding(),
                },
            ],
        });

        let bg_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG Shadow Tex BG"),
            layout: &bg_tex_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&shadow_view),
            }],
        });

        let shadow_blur_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Blur Horizontal BG"),
            layout: &shadow_blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&shadow_scratch_view),
                },
            ],
        });
        let shadow_blur_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Blur Vertical BG"),
            layout: &shadow_blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_scratch_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
            ],
        });

        let fluid_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Mesh Tex BG"),
            layout: &fluid_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bg_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&thickness_view),
                },
            ],
        });

        let wire_depth_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wireframe Depth BG"),
            layout: &wire_depth_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            }],
        });

        // ColorMap LUT
        let lut_data = build_color_lut_data();
        let color_lut_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Color LUT"),
            size: (lut_data.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&color_lut_buf, 0, bytemuck::cast_slice(&lut_data));

        // Resolve output buffers
        let out_bytes = (width * height * 4) as u64;
        let out_rgba_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Resolve Out RGBA Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Resolve Out Readback Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resolve BG"),
            layout: &resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_rgba_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            width,
            height,
            rgb_buffer: vec![0; width * height * 3],
            render_quality: quality,
            ssaa_factor: ssaa,
            show_wireframe: true,
            show_grid: true,
            env_preset: EnvironmentPreset::StudioDark,
            exposure: 1.0,
            particle_foam: true,
            lit_spheres: true,
            light_dir: Vec3::new(0.42, 0.82, 0.38).normalize(),
            light_intensity: 1.25,
            ambient_intensity: 0.32,
            device_name,
            render_width,
            render_height,
            shadow_res,
            device,
            queue,
            hdr_tex,
            hdr_view,
            bg_tex,
            thickness_view,
            depth_view,
            shadow_view,
            bg_pipeline,
            obstacle_pipeline,
            shadow_pipeline,
            shadow_blur_h_pipeline,
            shadow_blur_v_pipeline,
            thickness_pipeline,
            fluid_pipeline,
            particle_pipeline,
            wireframe_pipeline,
            resolve_pipeline,
            uniform_buf,
            collider_buf,
            global_bg,
            bg_tex_bg,
            shadow_blur_h_bg,
            shadow_blur_v_bg,
            fluid_tex_bg,
            wire_depth_bg,
            particle_bgl,
            color_lut_buf,
            particle_upload_buf: None,
            particle_upload_cap: 0,
            particle_upload_bg: None,
            cached_ext_particle_buf: None,
            cached_ext_particle_bg: None,
            out_rgba_buf,
            out_read_buf,
            resolve_bg,
            mesh_vbo: None,
            mesh_vbo_cap: 0,
            mesh_ibo: None,
            mesh_ibo_cap: 0,
            obstacle_vbo: None,
            obstacle_vbo_cap: 0,
            wire_vbo: None,
            wire_vbo_cap: 0,
            collider_scratch: Vec::with_capacity(MAX_ANALYTIC_COLLIDERS),
            particle_scratch: Vec::new(),
            obstacle_scratch: Vec::new(),
            wire_scratch: Vec::new(),
        })
    }

    pub fn internal_dimensions(&self) -> (usize, usize) {
        (self.render_width, self.render_height)
    }

    fn upload_colliders_and_obstacles(&mut self, colliders: &[Collider]) -> (u32, u32) {
        self.collider_scratch.clear();
        self.obstacle_scratch.clear();

        for col in colliders {
            if !col.enabled {
                continue;
            }
            match &col.shape {
                ColliderShape::Sphere { center, radius } => {
                    if self.collider_scratch.len() < MAX_ANALYTIC_COLLIDERS {
                        self.collider_scratch.push(GpuColliderItem {
                            min_or_center: [center.x, center.y, center.z, *radius],
                            max_or_ext: [0.0; 4],
                            kind_pad: [0, 0, 0, 0],
                        });
                    }
                }
                ColliderShape::Box { aabb } => {
                    if self.collider_scratch.len() < MAX_ANALYTIC_COLLIDERS {
                        self.collider_scratch.push(GpuColliderItem {
                            min_or_center: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
                            max_or_ext: [aabb.max.x, aabb.max.y, aabb.max.z, 0.0],
                            kind_pad: [1, 0, 0, 0],
                        });
                    }
                }
                ColliderShape::Cylinder {
                    center,
                    radius,
                    height,
                } => {
                    if self.collider_scratch.len() < MAX_ANALYTIC_COLLIDERS {
                        self.collider_scratch.push(GpuColliderItem {
                            min_or_center: [center.x, center.y, center.z, *radius],
                            max_or_ext: [*height, 0.0, 0.0, 0.0],
                            kind_pad: [2, 0, 0, 0],
                        });
                    }
                }
                ColliderShape::Mesh(tri_mesh) => {
                    let rot = Mat3::from_euler(
                        glam::EulerRot::YXZ,
                        tri_mesh.rotation.y,
                        tri_mesh.rotation.x,
                        tri_mesh.rotation.z,
                    );
                    let s = tri_mesh.scale;
                    let pos = tri_mesh.position;
                    for tri in tri_mesh.triangles.iter() {
                        let p0 = pos + rot * (tri.a * s);
                        let p1 = pos + rot * (tri.b * s);
                        let p2 = pos + rot * (tri.c * s);
                        let wn = (rot * tri.normal).normalize_or_zero().to_array();
                        self.obstacle_scratch.push(ObstacleVertex {
                            position: p0.to_array(),
                            normal: wn,
                        });
                        self.obstacle_scratch.push(ObstacleVertex {
                            position: p1.to_array(),
                            normal: wn,
                        });
                        self.obstacle_scratch.push(ObstacleVertex {
                            position: p2.to_array(),
                            normal: wn,
                        });
                    }
                }
            }
        }

        let n_cols = self.collider_scratch.len() as u32;
        if n_cols > 0 {
            self.queue.write_buffer(
                &self.collider_buf,
                0,
                bytemuck::cast_slice(&self.collider_scratch),
            );
        }

        let obs_verts = self.obstacle_scratch.len() as u32;
        if obs_verts > 0 {
            let need = self.obstacle_scratch.len();
            if need > self.obstacle_vbo_cap || self.obstacle_vbo.is_none() {
                let cap = need.max(256).next_power_of_two();
                self.obstacle_vbo = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GPU Obstacle VBO"),
                    size: (cap * std::mem::size_of::<ObstacleVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.obstacle_vbo_cap = cap;
            }
            self.queue.write_buffer(
                self.obstacle_vbo.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&self.obstacle_scratch),
            );
        }

        (n_cols, obs_verts)
    }

    fn upload_wireframe(&mut self, chamber: &Chamber, bounds: Aabb, is_2d: bool) -> u32 {
        self.wire_scratch.clear();
        let lines = if is_2d {
            chamber.generate_wireframe_lines_2d(&bounds)
        } else {
            chamber.generate_wireframe_lines(&bounds)
        };
        for (p0, p1) in lines {
            self.wire_scratch.push(LineVertex {
                position: p0.to_array(),
            });
            self.wire_scratch.push(LineVertex {
                position: p1.to_array(),
            });
        }
        let count = self.wire_scratch.len() as u32;
        if count > 0 {
            let need = self.wire_scratch.len();
            if need > self.wire_vbo_cap || self.wire_vbo.is_none() {
                let cap = need.max(256).next_power_of_two();
                self.wire_vbo = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GPU Wireframe VBO"),
                    size: (cap * std::mem::size_of::<LineVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.wire_vbo_cap = cap;
            }
            self.queue.write_buffer(
                self.wire_vbo.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&self.wire_scratch),
            );
        }
        count
    }

    #[allow(clippy::too_many_arguments)]
    fn build_base_uniforms(
        &self,
        camera: &Camera,
        bounds: Aabb,
        light_dir: Vec3,
        light_intensity: f32,
        ambient_intensity: f32,
        env_preset: EnvironmentPreset,
        exposure: f32,
        show_grid: bool,
        num_colliders: u32,
        is_2d: bool,
    ) -> GpuRenderUniforms {
        let aspect = self.render_width as f32 / self.render_height.max(1) as f32;
        let view = camera.view_matrix();
        let proj = Mat4::perspective_rh(camera.fov_y, aspect, camera.near, camera.far);
        let view_proj = proj * view;
        let eye_pos = camera.eye_position();
        let cam_fwd = {
            let f = (camera.target - eye_pos).normalize_or_zero();
            if f.length_squared() > 0.5 { f } else { Vec3::NEG_Z }
        };
        let cam_right = {
            let r = cam_fwd.cross(Vec3::Y).normalize_or_zero();
            if r.length_squared() > 0.5 { r } else { Vec3::X }
        };
        let cam_up = cam_right.cross(cam_fwd).normalize_or_zero();

        let wl = if light_dir.length_squared() > 0.25 {
            light_dir.normalize()
        } else {
            Vec3::new(0.42, 0.82, 0.38).normalize()
        };
        let eye_ld = (Mat3::from_mat4(view) * wl).normalize_or_zero();
        let eye_ld = if eye_ld.length_squared() > 0.25 {
            eye_ld
        } else {
            Vec3::new(0.42, 0.82, 0.38).normalize()
        };

        let floor_y = bounds.min.y - 0.002;
        let floor_margin = (bounds.size().x.max(bounds.size().z) * 0.85 + 0.8).max(1.6);

        let shadow_center = bounds.center();
        let half_ext = (bounds.size() * 0.5).max(Vec3::splat(0.5)) + Vec3::splat(0.8);
        let shadow_r = half_ext.length();
        let light_eye = shadow_center + wl * (shadow_r + 1.0);
        let s_up = if wl.y.abs() > 0.98 { Vec3::Z } else { Vec3::Y };
        let light_view = Mat4::look_at_rh(light_eye, shadow_center, s_up);
        let light_proj =
            Mat4::orthographic_rh(-shadow_r, shadow_r, -shadow_r, shadow_r, 0.1, shadow_r * 2.0 + 4.0);
        let shadow_vp = light_proj * light_view;

        let (bg_low, bg_high) = env_preset.bg_gradient_linear();
        let (sky_hor, sky_zen, sky_gnd) = env_preset.sky_reflection_linear();

        GpuRenderUniforms {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            view_proj: view_proj.to_cols_array_2d(),
            shadow_vp: shadow_vp.to_cols_array_2d(),
            eye_pos: [eye_pos.x, eye_pos.y, eye_pos.z, (camera.fov_y * 0.5).tan()],
            cam_fwd: [cam_fwd.x, cam_fwd.y, cam_fwd.z, aspect],
            cam_right: [cam_right.x, cam_right.y, cam_right.z, camera.near],
            cam_up: [cam_up.x, cam_up.y, cam_up.z, camera.far],
            light_dir: [wl.x, wl.y, wl.z, light_intensity],
            eye_light_dir: [eye_ld.x, eye_ld.y, eye_ld.z, ambient_intensity],
            bounds_min: [bounds.min.x, bounds.min.y, bounds.min.z, floor_y],
            bounds_max: [bounds.max.x, bounds.max.y, bounds.max.z, floor_margin],
            bg_low: [bg_low.x, bg_low.y, bg_low.z, exposure.max(0.05)],
            bg_high: [bg_high.x, bg_high.y, bg_high.z, 0.0],
            sky_horizon: [sky_hor.x, sky_hor.y, sky_hor.z, 0.02],
            sky_zenith: [sky_zen.x, sky_zen.y, sky_zen.z, 1000.0],
            sky_ground: [sky_gnd.x, sky_gnd.y, sky_gnd.z, 0.0],
            shallow_lin: [0.0, 0.0, 0.0, 0.62],
            deep_lin: [0.0, 0.0, 0.0, 2.1],
            shallow_srgb: [0.0, 0.0, 0.0, 1.333],
            deep_srgb: [0.0, 0.0, 0.0, 0.05],
            mat_opt1: [1.0, 0.6, 0.45, 0.9],
            mat_opt2: [0.0, 0.35, 6.0, 0.3],
            wire_color: [
                78.0 / 255.0,
                136.0 / 255.0,
                205.0 / 255.0,
                1.0 + self.ssaa_factor.saturating_sub(1) as f32 * 0.25,
            ],
            dims: [
                self.render_width as u32,
                self.render_height as u32,
                self.width as u32,
                self.height as u32,
            ],
            flags0: [
                self.ssaa_factor as u32,
                self.shadow_res as u32,
                num_colliders,
                if is_2d { 1 } else { 0 },
            ],
            flags1: [
                if show_grid { 1 } else { 0 },
                env_preset_id(env_preset),
                0,
                if self.lit_spheres { 1 } else { 0 },
            ],
            flags2: [
                0,
                0,
                if self.particle_foam { 1 } else { 0 },
                0,
            ],
        }
    }

    fn resolve_and_readback(&mut self, mut encoder: wgpu::CommandEncoder) {
        let empty_bgl = self
            .resolve_pipeline
            .get_bind_group_layout(1);
        let empty_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &empty_bgl,
            entries: &[],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU SSAA Resolve Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.resolve_pipeline);
            cpass.set_bind_group(0, &self.global_bg, &[]);
            cpass.set_bind_group(1, &empty_bg, &[]);
            cpass.set_bind_group(2, &self.resolve_bg, &[]);
            let wx = ((self.width + 15) / 16) as u32;
            let wy = ((self.height + 15) / 16) as u32;
            cpass.dispatch_workgroups(wx, wy, 1);
        }

        let copy_bytes = (self.width * self.height * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.out_rgba_buf, 0, &self.out_read_buf, 0, copy_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.out_read_buf.slice(0..copy_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let mapped = slice.get_mapped_range();
            let pixels: &[u32] = bytemuck::cast_slice(&mapped);
            self.rgb_buffer.resize(self.width * self.height * 3, 0);
            self.rgb_buffer
                .par_chunks_mut(self.width * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    let row_in = &pixels[y * self.width..(y + 1) * self.width];
                    for (x, &px) in row_in.iter().enumerate() {
                        let o = x * 3;
                        row[o] = (px & 0xff) as u8;
                        row[o + 1] = ((px >> 8) & 0xff) as u8;
                        row[o + 2] = ((px >> 16) & 0xff) as u8;
                    }
                });
            drop(mapped);
            self.out_read_buf.unmap();
        }
    }

    fn encode_bg_and_obstacles(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        obs_verts: u32,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("BG + Obstacles Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.bg_pipeline);
        rpass.set_bind_group(0, &self.global_bg, &[]);
        rpass.set_bind_group(1, &self.bg_tex_bg, &[]);
        rpass.draw(0..3, 0..1);

        if obs_verts > 0 {
            rpass.set_pipeline(&self.obstacle_pipeline);
            rpass.set_bind_group(0, &self.global_bg, &[]);
            rpass.set_vertex_buffer(0, self.obstacle_vbo.as_ref().unwrap().slice(..));
            rpass.draw(0..obs_verts, 0..1);
        }
    }

    fn encode_wireframe(&self, encoder: &mut wgpu::CommandEncoder, wire_verts: u32) {
        if wire_verts == 0 {
            return;
        }
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Wireframe Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.hdr_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&self.wireframe_pipeline);
        rpass.set_bind_group(0, &self.global_bg, &[]);
        rpass.set_bind_group(1, &self.wire_depth_bg, &[]);
        rpass.set_vertex_buffer(0, self.wire_vbo.as_ref().unwrap().slice(..));
        rpass.draw(0..wire_verts, 0..1);
    }

    /// Render particles directly from an external live `GpuSphSolver` particle storage buffer
    /// (zero CPU particle readback!).
    #[allow(clippy::too_many_arguments)]
    pub fn render_particles_from_gpu_buffer(
        &mut self,
        particle_buffer: &wgpu::Buffer,
        particle_count: usize,
        particle_radius: f32,
        chamber: &Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        camera: &Camera,
        color_mode: ColorMode,
        color_map: ColorMapType,
        rest_density: f32,
        is_2d: bool,
    ) {
        if self.cached_ext_particle_buf.as_ref() != Some(particle_buffer)
            || self.cached_ext_particle_bg.is_none()
        {
            self.cached_ext_particle_bg =
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("External Particle BG"),
                    layout: &self.particle_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: particle_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.color_lut_buf.as_entire_binding(),
                        },
                    ],
                }));
            self.cached_ext_particle_buf = Some(particle_buffer.clone());
        }
        let p_bg = self.cached_ext_particle_bg.as_ref().unwrap().clone();
        self.render_particles_with_bind_group(
            &p_bg,
            particle_count,
            particle_radius,
            chamber,
            bounds,
            colliders,
            camera,
            color_mode,
            color_map,
            rest_density,
            is_2d,
        );
    }

    /// Render particles supplied from CPU slice (uploads compact `GpuParticle` array to VRAM once).
    #[allow(clippy::too_many_arguments)]
    pub fn render_particles_studio(
        &mut self,
        particles: &[Particle],
        particle_radius: f32,
        chamber: &Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        camera: &Camera,
        color_mode: ColorMode,
        color_map: ColorMapType,
        rest_density: f32,
        is_2d: bool,
    ) {
        self.particle_scratch.clear();
        self.particle_scratch.extend(particles.iter().map(|p| GpuParticle {
            pos_type: [
                p.position.x,
                p.position.y,
                p.position.z,
                if p.particle_type == ParticleType::Boundary { 1.0 } else { 0.0 },
            ],
            vel_dens: [p.velocity.x, p.velocity.y, p.velocity.z, p.density],
        }));

        let count = self.particle_scratch.len();
        if count > 0 {
            if count > self.particle_upload_cap || self.particle_upload_buf.is_none() {
                let cap = count.max(1024).next_power_of_two();
                let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Uploaded Particle Storage"),
                    size: (cap * std::mem::size_of::<GpuParticle>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.particle_upload_bg =
                    Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Uploaded Particle BG"),
                        layout: &self.particle_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.color_lut_buf.as_entire_binding(),
                            },
                        ],
                    }));
                self.particle_upload_buf = Some(buf);
                self.particle_upload_cap = cap;
            }
            self.queue.write_buffer(
                self.particle_upload_buf.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&self.particle_scratch),
            );
        } else if self.particle_upload_buf.is_none() {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Empty Particle Storage"),
                size: std::mem::size_of::<GpuParticle>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.particle_upload_bg =
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Empty Particle BG"),
                    layout: &self.particle_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.color_lut_buf.as_entire_binding(),
                        },
                    ],
                }));
            self.particle_upload_buf = Some(buf);
            self.particle_upload_cap = 1;
        }

        let p_bg = self.particle_upload_bg.as_ref().unwrap().clone();
        self.render_particles_with_bind_group(
            &p_bg,
            count,
            particle_radius,
            chamber,
            bounds,
            colliders,
            camera,
            color_mode,
            color_map,
            rest_density,
            is_2d,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn render_particles_with_bind_group(
        &mut self,
        particle_bg: &wgpu::BindGroup,
        particle_count: usize,
        particle_radius: f32,
        chamber: &Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        camera: &Camera,
        color_mode: ColorMode,
        color_map: ColorMapType,
        rest_density: f32,
        is_2d: bool,
    ) {
        let (n_cols, obs_verts) = self.upload_colliders_and_obstacles(colliders);
        let wire_verts = if self.show_wireframe {
            self.upload_wireframe(chamber, bounds, is_2d)
        } else {
            0
        };

        let mut uniforms = self.build_base_uniforms(
            camera,
            bounds,
            self.light_dir,
            self.light_intensity,
            self.ambient_intensity,
            self.env_preset,
            self.exposure,
            self.show_grid,
            n_cols,
            is_2d,
        );
        uniforms.sky_horizon[3] = particle_radius.max(1e-5);
        uniforms.sky_zenith[3] = rest_density.max(1e-4);
        uniforms.flags2[0] = color_mode_id(color_mode);
        uniforms.flags2[1] = color_map_id(color_map);
        uniforms.flags2[2] = if self.particle_foam { 1 } else { 0 };
        uniforms.flags2[3] = 0;

        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU Particle Frame Encoder"),
            });

        self.encode_bg_and_obstacles(&mut encoder, obs_verts);

        if particle_count > 0 {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.particle_pipeline);
            rpass.set_bind_group(0, &self.global_bg, &[]);
            rpass.set_bind_group(1, particle_bg, &[]);
            rpass.draw(0..6, 0..particle_count as u32);
        }

        if self.show_wireframe {
            self.encode_wireframe(&mut encoder, wire_verts);
        }

        self.resolve_and_readback(encoder);
    }

    /// Render reconstructed `SurfaceMesh` on GPU with directional volumetric floor shadows & caustics,
    /// camera-space optical thickness Beer-Lambert absorption, chromatic refraction, GGX specular,
    /// studio reflections, procedural textures, and depth-aware wireframes.
    #[allow(clippy::too_many_arguments)]
    pub fn render_surface_mesh_frame(
        &mut self,
        mesh: &SurfaceMesh,
        cfg: &SurfaceMeshConfig,
        camera: &Camera,
        chamber: &Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        anim_time: f32,
        is_2d: bool,
    ) {
        let (n_cols, obs_verts) = self.upload_colliders_and_obstacles(colliders);
        let wire_verts = if cfg.show_wireframe {
            self.upload_wireframe(chamber, bounds, is_2d)
        } else {
            0
        };

        let mut uniforms = self.build_base_uniforms(
            camera,
            bounds,
            cfg.light_dir,
            cfg.light_intensity,
            cfg.ambient_intensity,
            cfg.env_preset,
            cfg.exposure,
            cfg.show_grid,
            n_cols,
            is_2d,
        );

        let shallow_srgb = Vec3::from_array(cfg.shallow_color);
        let deep_srgb = Vec3::from_array(cfg.deep_color);
        let shallow_lin = srgb_slice_to_linear(cfg.shallow_color);
        let deep_lin = srgb_slice_to_linear(cfg.deep_color);
        let has_shadow_fluid =
            cfg.cast_shadows && !mesh.is_empty() && uniforms.light_dir[1] > 0.08;

        uniforms.bg_high[3] = anim_time;
        uniforms.sky_ground[3] = self.render_quality.chromatic_dispersion();
        uniforms.shallow_lin = [shallow_lin.x, shallow_lin.y, shallow_lin.z, cfg.opacity];
        uniforms.deep_lin = [deep_lin.x, deep_lin.y, deep_lin.z, cfg.absorption];
        uniforms.shallow_srgb = [shallow_srgb.x, shallow_srgb.y, shallow_srgb.z, cfg.ior];
        uniforms.deep_srgb = [deep_srgb.x, deep_srgb.y, deep_srgb.z, cfg.roughness];
        uniforms.mat_opt1 = [
            cfg.specular_intensity,
            cfg.refraction_strength,
            cfg.subsurface_scattering,
            cfg.env_reflection,
        ];
        uniforms.mat_opt2 = [
            cfg.metallic,
            cfg.foam_intensity,
            cfg.texture_scale,
            cfg.texture_strength,
        ];
        uniforms.flags1[2] = cfg.texture_mode.gpu_id() as u32;
        uniforms.flags2[3] = if has_shadow_fluid { 1 } else { 0 };

        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        if !mesh.is_empty() {
            let v_len = mesh.vertices.len();
            if v_len > self.mesh_vbo_cap || self.mesh_vbo.is_none() {
                let cap = v_len.max(1024).next_power_of_two();
                self.mesh_vbo = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GPU Surface Mesh VBO"),
                    size: (cap * std::mem::size_of::<MeshVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.mesh_vbo_cap = cap;
            }
            let i_len = mesh.indices.len();
            if i_len > self.mesh_ibo_cap || self.mesh_ibo.is_none() {
                let cap = i_len.max(3072).next_power_of_two();
                self.mesh_ibo = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GPU Surface Mesh IBO"),
                    size: (cap * std::mem::size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.mesh_ibo_cap = cap;
            }
            self.queue.write_buffer(
                self.mesh_vbo.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&mesh.vertices),
            );
            self.queue.write_buffer(
                self.mesh_ibo.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&mesh.indices),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU Surface Mesh Frame Encoder"),
            });

        if has_shadow_fluid {
            {
                let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GPU Shadow Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.shadow_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                spass.set_pipeline(&self.shadow_pipeline);
                spass.set_bind_group(0, &self.global_bg, &[]);
                spass.set_vertex_buffer(0, self.mesh_vbo.as_ref().unwrap().slice(..));
                spass.set_index_buffer(
                    self.mesh_ibo.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                spass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
            }

            let blur_rounds = if self.shadow_res >= 256 { 2 } else { 1 };
            let wg = (self.shadow_res as u32).div_ceil(16);
            for _ in 0..blur_rounds {
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("GPU Shadow Blur Horizontal"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.shadow_blur_h_pipeline);
                    cpass.set_bind_group(0, &self.global_bg, &[]);
                    cpass.set_bind_group(1, &self.shadow_blur_h_bg, &[]);
                    cpass.dispatch_workgroups(wg, wg, 1);
                }
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("GPU Shadow Blur Vertical"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.shadow_blur_v_pipeline);
                    cpass.set_bind_group(0, &self.global_bg, &[]);
                    cpass.set_bind_group(1, &self.shadow_blur_v_bg, &[]);
                    cpass.dispatch_workgroups(wg, wg, 1);
                }
            }
        }

        self.encode_bg_and_obstacles(&mut encoder, obs_verts);

        if !mesh.is_empty() {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.hdr_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.bg_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.render_width as u32,
                    height: self.render_height as u32,
                    depth_or_array_layers: 1,
                },
            );

            // Volumetric fluid thickness pass
            {
                let mut tpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GPU Thickness Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.thickness_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                tpass.set_pipeline(&self.thickness_pipeline);
                tpass.set_bind_group(0, &self.global_bg, &[]);
                tpass.set_vertex_buffer(0, self.mesh_vbo.as_ref().unwrap().slice(..));
                tpass.set_index_buffer(
                    self.mesh_ibo.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                tpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
            }

            // Physically-based liquid PBR surface pass
            {
                let mut fpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GPU Fluid Mesh Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                fpass.set_pipeline(&self.fluid_pipeline);
                fpass.set_bind_group(0, &self.global_bg, &[]);
                fpass.set_bind_group(1, &self.fluid_tex_bg, &[]);
                fpass.set_vertex_buffer(0, self.mesh_vbo.as_ref().unwrap().slice(..));
                fpass.set_index_buffer(
                    self.mesh_ibo.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                fpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
            }
        }

        if cfg.show_wireframe {
            self.encode_wireframe(&mut encoder, wire_verts);
        }

        self.resolve_and_readback(encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{FluidMaterialPreset, SurfaceMeshConfig};
    use crate::sph::chamber::Chamber;

    #[test]
    fn gpu_renderer_renders_particles_and_surface_mesh_presets() {
        let Ok(mut rend) = GpuRenderer::new_standalone(160, 120, CliRenderQuality::Studio) else {
            return;
        };

        let mut particles = Vec::new();
        let r = 0.03f32;
        let spacing = r * 1.9;
        for z in 0..6 {
            for y in 0..6 {
                for x in 0..6 {
                    let mut p = Particle::new(
                        Vec3::new(
                            -0.15 + x as f32 * spacing,
                            0.05 + y as f32 * spacing,
                            -0.15 + z as f32 * spacing,
                        ),
                        Vec3::new(0.6, 0.2, -0.3),
                        1000.0,
                        particles.len() as u32,
                    );
                    p.particle_type = ParticleType::Fluid;
                    particles.push(p);
                }
            }
        }

        let bounds = Aabb::new(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 0.8, 0.5));
        let chamber = Chamber::default();
        let cam = Camera::default();

        // 1. Particle studio render (SSAA 2x)
        rend.render_particles_studio(
            &particles,
            r,
            &chamber,
            bounds,
            &[],
            &cam,
            ColorMode::Velocity,
            ColorMapType::Ocean,
            1000.0,
            false,
        );
        assert_eq!(rend.rgb_buffer.len(), 160 * 120 * 3);
        let non_zero_p = rend.rgb_buffer.iter().filter(|&&b| b > 20).count();
        assert!(non_zero_p > 500, "GPU particle frame should contain lit pixels");

        // 2. SurfaceMesh ClearWater + LiquidChrome (Mercury)
        let mut cfg = SurfaceMeshConfig::default();
        let mesh = SurfaceMesh::from_particles(&particles, r, r * 2.5, &cfg);
        assert!(!mesh.is_empty());

        FluidMaterialPreset::ClearWater.apply(&mut cfg);
        rend.render_surface_mesh_frame(&mesh, &cfg, &cam, &chamber, bounds, &[], 0.5, false);
        let non_zero_w = rend.rgb_buffer.iter().filter(|&&b| b > 30).count();
        assert!(non_zero_w > 500, "GPU ClearWater frame should contain lit pixels");

        FluidMaterialPreset::LiquidChrome.apply(&mut cfg);
        rend.render_surface_mesh_frame(&mesh, &cfg, &cam, &chamber, bounds, &[], 0.5, false);
        let max_px = *rend.rgb_buffer.iter().max().unwrap_or(&0);
        assert!(max_px > 120, "GPU LiquidChrome frame should contain specular reflections");
    }
}

