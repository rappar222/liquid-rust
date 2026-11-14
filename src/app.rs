use crate::export::{Mp4Encoder, PlyExporter, PngExporter, SoftwareRenderer};
use crate::gpu::{create_headless_wgpu_device, GpuRenderer, GpuSphSolver, GpuSurfaceField};
use crate::math::{unproject_mouse_ray, Aabb};
use crate::render::{
    Camera, CliRenderQuality, ColorMap, ColorMapType, ColorMode, EnvironmentPreset,
    FluidMaterialPreset, GlRenderer, MeshTextureMode, ParticleVertex, RenderMode, SurfaceMesh,
    SurfaceMeshConfig,
};
use crate::scene_io::{
    CameraDto, ChamberDto, ColliderDto, DrainDto, FaucetDto, SceneConfigFile, SolverConfigDto,
    SurfaceMeshDto, VisualsDto,
};
use crate::sph::scene::{SceneGenerator, ScenePreset};
use crate::sph::solver::{
    compute_lattice_rest_density, compute_lattice_rest_density_2d, FluidPhysicsPreset,
    SimulationQuality, SolverConfig, SphSolver,
};
use crate::sph::{
    Chamber, ChamberShape, Collider, ColliderShape, CubeFace, Drain, Faucet, FaucetShape, Particle,
};
use crate::updater::{
    cleanup_old_binary, format_bytes, restart_into_updated_binary, UpdatePhase, UpdaterHandle,
    CURRENT_VERSION,
};
use eframe::egui;
use glam::Vec3;
use glow::HasContext as _;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliCommandMode {
    Gui,
    Mp4Video,
    PngFrame,
    Headless,
    PlySequence,
}

impl CliCommandMode {
    pub const ALL: &'static [CliCommandMode] = &[
        CliCommandMode::Gui,
        CliCommandMode::Mp4Video,
        CliCommandMode::PngFrame,
        CliCommandMode::Headless,
        CliCommandMode::PlySequence,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            CliCommandMode::Gui => "Interactive GUI",
            CliCommandMode::Mp4Video => "MP4 Video",
            CliCommandMode::PngFrame => "PNG Frame",
            CliCommandMode::Headless => "Headless Sim",
            CliCommandMode::PlySequence => "PLY Sequence",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InspectorTab {
    Scene,
    Physics,
    Surface,
    Render,
    RenderCli,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CentralView {
    LiveViewport,
    RenderResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StudioRenderKind {
    StillFrame,
    Animation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationStartSource {
    PresetStart,
    CurrentState,
}

/// Visual representation used specifically for offline Studio renders (can differ from the live 3D editor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderAppearance {
    MatchViewport,
    SmoothLiquid,
    ParticleSpheres,
}

/// One-click beginner-friendly studio render presets bringing full CLI rendering power to the GUI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuidedRenderPreset {
    QuickPreview,
    RecommendedHd,
    Cinematic2K,
    Ultra4K,
    StudioPhoto,
    Custom,
}

impl GuidedRenderPreset {
    pub const ALL_PRESETS: &'static [GuidedRenderPreset] = &[
        GuidedRenderPreset::QuickPreview,
        GuidedRenderPreset::RecommendedHd,
        GuidedRenderPreset::Cinematic2K,
        GuidedRenderPreset::Ultra4K,
        GuidedRenderPreset::StudioPhoto,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            GuidedRenderPreset::QuickPreview => "Quick Draft",
            GuidedRenderPreset::RecommendedHd => "Full HD Studio ★",
            GuidedRenderPreset::Cinematic2K => "Cinema 2K",
            GuidedRenderPreset::Ultra4K => "4K Ultra VFX",
            GuidedRenderPreset::StudioPhoto => "High-Res Photo",
            GuidedRenderPreset::Custom => "Custom Setup",
        }
    }

    pub fn summary(&self) -> &'static str {
        match self {
            GuidedRenderPreset::QuickPreview => {
                "Fast 720p • 30 FPS video (~18k particles) — great for checking camera & motion in seconds."
            }
            GuidedRenderPreset::RecommendedHd => {
                "Recommended: 1080p • 60 FPS smooth liquid video (~50k particles) with soft shadows & reflections."
            }
            GuidedRenderPreset::Cinematic2K => {
                "2K 1440p • 60 FPS high-detail movie (~120k particles) with 3x cinema anti-aliasing."
            }
            GuidedRenderPreset::Ultra4K => {
                "4K UHD • 60 FPS ultra-dense VFX (~1M particles) — maximum quality, takes several minutes."
            }
            GuidedRenderPreset::StudioPhoto => {
                "Crisp 2K 1440p still photograph (~120k particles) with 3x anti-aliasing & soft studio shadows."
            }
            GuidedRenderPreset::Custom => {
                "Customized offline render settings configured below."
            }
        }
    }
}

/// Crisp geometric vector icons painted directly via `egui::Painter` so every button, tab,
/// and status badge renders cleanly across all Linux/macOS/Windows fonts without missing glyphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UiIcon {
    Wave,
    Cube3D,
    Plane2D,
    Play,
    Pause,
    Step,
    StepBack,
    Reset,
    Camera,
    Spheres,
    Mesh,
    Sliders,
    Film,
    Copy,
    Folder,
    Save,
    Image,
    Help,
    Warning,
    Bolt,
}

/// Packed RGB8 frame stored for in-GUI render result display, animation scrubbing, and lossless PNG saving.
#[derive(Clone)]
pub struct PreviewFrameRgb {
    pub width: usize,
    pub height: usize,
    pub rgb: Arc<Vec<u8>>,
}

/// Shared progress & output state for non-blocking studio still & animation rendering in the GUI.
pub struct BgRenderState {
    pub is_running: bool,
    pub kind: StudioRenderKind,
    pub completed_frames: usize,
    pub total_frames: usize,
    pub elapsed_secs: f32,
    pub phase_label: String,
    pub sim_particles_count: usize,
    pub backend_label: String,
    pub output_width: usize,
    pub output_height: usize,
    pub quality: CliRenderQuality,
    pub fps: u32,
    pub saved_path: Option<String>,
    pub finished_message: Option<String>,
    pub full_res_frame: Option<PreviewFrameRgb>,
    pub anim_frames: Vec<PreviewFrameRgb>,
    pub frame_revision: u64,
    pub anim_generation: u64,
    pub just_finished_anim: bool,
}

/// Hardware Vulkan GPU offline renderer with automatic multi-core CPU SSAA fallback.
enum GuiOfflineRenderer {
    Gpu {
        renderer: GpuRenderer,
        surface_field: Option<GpuSurfaceField>,
    },
    Cpu(SoftwareRenderer),
}

impl GuiOfflineRenderer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        width: usize,
        height: usize,
        rq: CliRenderQuality,
        use_gpu: bool,
        shared_gpu: Option<&GpuSphSolver>,
        enable_culling: bool,
        lit_spheres: bool,
        particle_foam: bool,
        mesh_cfg: &SurfaceMeshConfig,
    ) -> Self {
        if use_gpu {
            let gpu_try = if let Some(gs) = shared_gpu {
                GpuRenderer::new_with_device(
                    Arc::clone(&gs.device),
                    Arc::clone(&gs.queue),
                    gs.device_name.clone(),
                    width,
                    height,
                    rq,
                )
                .ok()
                .map(|r| {
                    let sf =
                        GpuSurfaceField::new(Arc::clone(&gs.device), Arc::clone(&gs.queue)).ok();
                    (r, sf)
                })
            } else {
                create_headless_wgpu_device()
                    .ok()
                    .and_then(|(dev, queue, name)| {
                        let sf = GpuSurfaceField::new(Arc::clone(&dev), Arc::clone(&queue)).ok();
                        GpuRenderer::new_with_device(dev, queue, name, width, height, rq)
                            .ok()
                            .map(|r| (r, sf))
                    })
            };

            if let Some((mut r, sf)) = gpu_try {
                r.lit_spheres = lit_spheres;
                r.particle_foam = particle_foam;
                r.env_preset = mesh_cfg.env_preset;
                r.exposure = mesh_cfg.exposure;
                r.show_grid = mesh_cfg.show_grid;
                r.show_wireframe = mesh_cfg.show_wireframe;
                r.light_dir = mesh_cfg.light_dir;
                r.light_intensity = mesh_cfg.light_intensity;
                r.ambient_intensity = mesh_cfg.ambient_intensity;
                return Self::Gpu {
                    renderer: r,
                    surface_field: sf,
                };
            }
        }

        let mut sw = SoftwareRenderer::new_with_quality(width, height, rq);
        sw.enable_culling = enable_culling;
        sw.lit_spheres = lit_spheres;
        sw.particle_foam = particle_foam;
        sw.env_preset = mesh_cfg.env_preset;
        sw.exposure = mesh_cfg.exposure;
        sw.show_grid = mesh_cfg.show_grid;
        sw.show_wireframe = mesh_cfg.show_wireframe;
        sw.light_dir = mesh_cfg.light_dir;
        sw.light_intensity = mesh_cfg.light_intensity;
        sw.ambient_intensity = mesh_cfg.ambient_intensity;
        Self::Cpu(sw)
    }

    fn backend_label(&self) -> String {
        match self {
            Self::Gpu { renderer, .. } => {
                format!("Vulkan GPU ({}x SSAA)", renderer.ssaa_factor)
            }
            Self::Cpu(sw) => format!("CPU Raster ({}x SSAA)", sw.ssaa_factor),
        }
    }

    fn current_rgb(&self) -> &[u8] {
        match self {
            Self::Gpu { renderer, .. } => &renderer.rgb_buffer,
            Self::Cpu(sw) => &sw.rgb_buffer,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_frame(
        &mut self,
        render_mode: RenderMode,
        particles: &[Particle],
        gpu_solver: Option<&GpuSphSolver>,
        particle_radius: f32,
        smoothing_length: f32,
        display_size_factor: f32,
        rest_density: f32,
        mesh_cfg: &SurfaceMeshConfig,
        camera: &Camera,
        chamber: &Chamber,
        bounds: Aabb,
        colliders: &[Collider],
        color_mode: ColorMode,
        color_map: ColorMapType,
        anim_time: f32,
        is_2d: bool,
    ) -> &[u8] {
        match render_mode {
            RenderMode::Particles => {
                let rad = particle_radius * display_size_factor;
                match self {
                    Self::Gpu { renderer, .. } => {
                        if let Some(gs) = gpu_solver {
                            if gs.supports_vertex_storage() {
                                if let Some(buf) = gs.particle_buffer() {
                                    renderer.render_particles_from_gpu_buffer(
                                        buf,
                                        gs.particle_count,
                                        rad,
                                        chamber,
                                        bounds,
                                        colliders,
                                        camera,
                                        color_mode,
                                        color_map,
                                        rest_density,
                                        is_2d,
                                    );
                                    return &renderer.rgb_buffer;
                                }
                            }
                        }
                        renderer.render_particles_studio(
                            particles,
                            rad,
                            chamber,
                            bounds,
                            colliders,
                            camera,
                            color_mode,
                            color_map,
                            rest_density,
                            is_2d,
                        );
                        &renderer.rgb_buffer
                    }
                    Self::Cpu(sw) => {
                        sw.render_particles_studio(
                            particles,
                            rad,
                            chamber,
                            bounds,
                            colliders,
                            camera,
                            color_mode,
                            color_map,
                            rest_density,
                            is_2d,
                        );
                        &sw.rgb_buffer
                    }
                }
            }
            RenderMode::SurfaceMesh => match self {
                Self::Gpu {
                    renderer,
                    surface_field,
                } => {
                    let mesh = if let Some(ref mut sf) = surface_field {
                        SurfaceMesh::from_particles_with_gpu(
                            particles,
                            particle_radius,
                            smoothing_length,
                            mesh_cfg,
                            sf,
                        )
                    } else {
                        SurfaceMesh::from_particles(
                            particles,
                            particle_radius,
                            smoothing_length,
                            mesh_cfg,
                        )
                    };
                    renderer.render_surface_mesh_frame(
                        &mesh,
                        mesh_cfg,
                        camera,
                        chamber,
                        bounds,
                        colliders,
                        anim_time,
                        is_2d,
                    );
                    &renderer.rgb_buffer
                }
                Self::Cpu(sw) => {
                    let mesh = SurfaceMesh::from_particles(
                        particles,
                        particle_radius,
                        smoothing_length,
                        mesh_cfg,
                    );
                    sw.render_surface_mesh_frame(
                        &mesh,
                        mesh_cfg,
                        camera,
                        chamber,
                        bounds,
                        colliders,
                        anim_time,
                        is_2d,
                    );
                    &sw.rgb_buffer
                }
            },
        }
    }
}

pub struct LiquidApp {
    pub solver: SphSolver,
    pub gpu_solver: Option<GpuSphSolver>,
    pub use_gpu: bool,
    pub gpu_device_name: String,
    pub camera: Camera,
    pub camera_orbit_speed_deg: f32,
    pub is_playing: bool,
    pub fixed_dt: f32,
    pub sim_speed: f32,
    pub current_preset: ScenePreset,
    pub current_quality: SimulationQuality,
    pub fluid_preset: FluidPhysicsPreset,

    // Rendering & Surface Meshing settings
    pub renderer: Arc<Mutex<Option<GlRenderer>>>,
    pub render_mode: RenderMode,
    pub mesh_config: SurfaceMeshConfig,
    pub cached_surface_mesh: Arc<SurfaceMesh>,
    pub mesh_dirty: bool,
    pub last_mesh_time_ms: f32,
    pub sim_time: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,

    pub lit_spheres: bool,
    pub particle_foam: bool,
    pub enable_culling: bool,
    pub drawn_particles_count: usize,
    pub color_mode: ColorMode,
    pub color_map: ColorMapType,
    pub display_size_factor: f32,
    pub sphere_obstacle_pos: Vec3,
    pub sphere_obstacle_radius: f32,
    pub sphere_obstacle_enabled: bool,

    // Persistent vertex buffer to avoid per-frame heap allocations
    render_vertices: Vec<ParticleVertex>,

    // Diagnostics
    pub last_frame_time: Instant,
    pub last_camera_tick: Instant,
    pub frame_counter: usize,
    pub current_fps: f32,
    pub last_step_time_ms: f32,

    // Inspector & Studio Render / Export / CLI Generator
    pub active_tab: InspectorTab,
    pub central_view: CentralView,
    pub guided_render_preset: GuidedRenderPreset,
    pub render_kind: StudioRenderKind,
    pub render_appearance: RenderAppearance,
    pub render_sim_quality: Option<SimulationQuality>,
    pub render_still_simulate: bool,
    pub render_still_time: f32,
    pub render_also_export_obj: bool,
    pub render_also_export_ply: bool,
    pub render_anim_start: AnimationStartSource,
    pub render_save_mp4: bool,
    pub render_auto_save_png: bool,
    pub render_browse_pending: Arc<AtomicBool>,
    pub render_mp4_browse_result: Arc<Mutex<Option<String>>>,
    pub render_png_browse_result: Arc<Mutex<Option<String>>>,
    pub render_save_result_png_slot: Arc<Mutex<Option<String>>>,
    render_texture: Option<egui::TextureHandle>,
    render_uploaded_key: (u64, Option<usize>),
    pub render_zoom: f32,
    pub render_pan: egui::Vec2,
    pub anim_playing: bool,
    pub anim_loop: bool,
    pub anim_speed: f32,
    pub anim_play_idx: usize,
    anim_accum_secs: f32,
    last_anim_tick: Instant,
    pub theme_applied: bool,
    pub export_path: String,
    pub png_output_path: String,
    pub scene_file_path: String,
    pub export_sequence: bool,
    pub exported_frames_count: usize,
    pub status_message: String,
    pub show_extreme_modal: bool,
    pub show_shortcuts_modal: bool,
    pub obj_file_path: String,
    pub obj_browse_pending: Arc<AtomicBool>,
    pub obj_browse_result: Arc<Mutex<Option<String>>>,
    pub scene_browse_pending: Arc<AtomicBool>,
    pub scene_load_result: Arc<Mutex<Option<String>>>,
    pub scene_save_result: Arc<Mutex<Option<String>>>,
    pub next_collider_id: u32,
    pub next_faucet_id: u32,
    pub next_drain_id: u32,
    pub is_2d: bool,
    pub cli_mode: CliCommandMode,
    pub cli_duration: f32,
    pub cli_fps: u32,
    pub cli_mp4_output: String,
    pub gui_render_quality: CliRenderQuality,
    pub video_width: usize,
    pub video_height: usize,

    // Background MP4 render worker state
    pub bg_render: Arc<Mutex<BgRenderState>>,
    pub bg_render_cancel: Arc<AtomicBool>,

    // GitHub Release self-updater state
    pub updater: UpdaterHandle,
}

impl LiquidApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cleanup_old_binary();
        Self::apply_studio_theme(&cc.egui_ctx);

        let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
        let config = SolverConfig::default();
        let mut solver = SphSolver::new(bounds, config);

        let preset = ScenePreset::DamBreak;
        SceneGenerator::setup_scene(&mut solver, preset);

        let renderer = if let Some(ref gl) = cc.gl {
            match GlRenderer::new(gl) {
                Ok(r) => Arc::new(Mutex::new(Some(r))),
                Err(e) => {
                    eprintln!("Failed to initialize OpenGL renderer: {}", e);
                    Arc::new(Mutex::new(None))
                }
            }
        } else {
            Arc::new(Mutex::new(None))
        };

        let (gpu_solver, gpu_device_name, use_gpu) =
            match GpuSphSolver::new(bounds, &solver.config, &solver.particles) {
                Ok(g) => {
                    let name = g.device_name.clone();
                    (Some(g), name, true)
                }
                Err(e) => {
                    eprintln!("GPU solver init for GUI: {}", e);
                    (None, "GPU unavailable".to_string(), false)
                }
            };

        let mut app = Self::build_base(
            solver,
            gpu_solver,
            use_gpu,
            gpu_device_name,
            renderer,
            preset,
            true,
        );
        app.apply_guided_render_preset(GuidedRenderPreset::RecommendedHd);
        app.render_auto_save_png = true;
        app.theme_applied = true;
        app.updater
            .spawn_check(Some(cc.egui_ctx.clone()), false);
        app
    }

    pub fn new_headless() -> Self {
        let bounds = Aabb::new(Vec3::new(-0.8, 0.0, -0.45), Vec3::new(0.8, 1.2, 0.45));
        let config = SolverConfig::default();
        let mut solver = SphSolver::new(bounds, config);

        let preset = ScenePreset::DamBreak;
        SceneGenerator::setup_scene(&mut solver, preset);

        Self::build_base(
            solver,
            None,
            false,
            "Headless".to_string(),
            Arc::new(Mutex::new(None)),
            preset,
            false,
        )
    }

    fn build_base(
        solver: SphSolver,
        gpu_solver: Option<GpuSphSolver>,
        use_gpu: bool,
        gpu_device_name: String,
        renderer: Arc<Mutex<Option<GlRenderer>>>,
        preset: ScenePreset,
        is_playing: bool,
    ) -> Self {
        let mesh_config = SurfaceMeshConfig::default();
        let (az, el) = Self::dir_to_angles(mesh_config.light_dir);
        let init_cap = solver.particles.len();
        let now = Instant::now();

        Self {
            solver,
            gpu_solver,
            use_gpu,
            gpu_device_name,
            camera: Camera::default(),
            camera_orbit_speed_deg: 0.0,
            is_playing,
            fixed_dt: 0.016,
            sim_speed: 1.0,
            current_preset: preset,
            current_quality: SimulationQuality::Medium,
            fluid_preset: FluidPhysicsPreset::Water,
            renderer,
            render_mode: RenderMode::Particles,
            mesh_config,
            cached_surface_mesh: Arc::new(SurfaceMesh::default()),
            mesh_dirty: true,
            last_mesh_time_ms: 0.0,
            sim_time: 0.0,
            sun_azimuth_deg: az,
            sun_elevation_deg: el,
            lit_spheres: true,
            particle_foam: true,
            enable_culling: true,
            drawn_particles_count: 0,
            color_mode: ColorMode::Velocity,
            color_map: ColorMapType::Ocean,
            display_size_factor: 1.25,
            sphere_obstacle_pos: Vec3::new(0.0, 0.45, 0.0),
            sphere_obstacle_radius: 0.22,
            sphere_obstacle_enabled: false,
            render_vertices: Vec::with_capacity(init_cap),
            last_frame_time: now,
            last_camera_tick: now,
            frame_counter: 0,
            current_fps: 60.0,
            last_step_time_ms: 0.0,
            active_tab: InspectorTab::Scene,
            central_view: CentralView::LiveViewport,
            guided_render_preset: GuidedRenderPreset::Custom,
            render_kind: StudioRenderKind::Animation,
            render_appearance: RenderAppearance::MatchViewport,
            render_sim_quality: None,
            render_still_simulate: false,
            render_still_time: 1.5,
            render_also_export_obj: false,
            render_also_export_ply: false,
            render_anim_start: AnimationStartSource::PresetStart,
            render_save_mp4: true,
            render_auto_save_png: false,
            render_browse_pending: Arc::new(AtomicBool::new(false)),
            render_mp4_browse_result: Arc::new(Mutex::new(None)),
            render_png_browse_result: Arc::new(Mutex::new(None)),
            render_save_result_png_slot: Arc::new(Mutex::new(None)),
            render_texture: None,
            render_uploaded_key: (0, None),
            render_zoom: 0.0,
            render_pan: egui::Vec2::ZERO,
            anim_playing: false,
            anim_loop: true,
            anim_speed: 1.0,
            anim_play_idx: 0,
            anim_accum_secs: 0.0,
            last_anim_tick: now,
            theme_applied: false,
            export_path: "exports".to_string(),
            png_output_path: "snapshot.png".to_string(),
            scene_file_path: "scene.json".to_string(),
            export_sequence: false,
            exported_frames_count: 0,
            status_message: "Ready".to_string(),
            show_extreme_modal: false,
            show_shortcuts_modal: false,
            obj_file_path: "model.obj".to_string(),
            obj_browse_pending: Arc::new(AtomicBool::new(false)),
            obj_browse_result: Arc::new(Mutex::new(None)),
            scene_browse_pending: Arc::new(AtomicBool::new(false)),
            scene_load_result: Arc::new(Mutex::new(None)),
            scene_save_result: Arc::new(Mutex::new(None)),
            next_collider_id: 1,
            next_faucet_id: 1,
            next_drain_id: 1,
            is_2d: false,
            cli_mode: CliCommandMode::Gui,
            cli_duration: 5.0,
            cli_fps: 60,
            cli_mp4_output: "simulation.mp4".to_string(),
            gui_render_quality: CliRenderQuality::Studio,
            video_width: 1280,
            video_height: 720,
            bg_render: Arc::new(Mutex::new(BgRenderState {
                is_running: false,
                kind: StudioRenderKind::Animation,
                completed_frames: 0,
                total_frames: 0,
                elapsed_secs: 0.0,
                phase_label: String::new(),
                sim_particles_count: 0,
                backend_label: String::new(),
                output_width: 1280,
                output_height: 720,
                quality: CliRenderQuality::Studio,
                fps: 60,
                saved_path: None,
                finished_message: None,
                full_res_frame: None,
                anim_frames: Vec::new(),
                frame_revision: 0,
                anim_generation: 0,
                just_finished_anim: false,
            })),
            bg_render_cancel: Arc::new(AtomicBool::new(false)),
            updater: UpdaterHandle::new(),
        }
    }

    /// Apply sleek dark engineering studio theme to `egui`.
    pub fn apply_studio_theme(ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(10, 14, 22);
        visuals.window_fill = egui::Color32::from_rgb(13, 18, 28);
        visuals.extreme_bg_color = egui::Color32::from_rgb(7, 10, 16);
        visuals.faint_bg_color = egui::Color32::from_rgb(16, 22, 34);
        visuals.code_bg_color = egui::Color32::from_rgb(8, 12, 20);

        let border_subtle = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(33, 45, 67));
        let border_hover = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(68, 142, 230));
        let accent_fill = egui::Color32::from_rgb(20, 110, 200);
        let accent_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(82, 182, 255));

        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(14, 19, 30);
        visuals.widgets.noninteractive.bg_stroke = border_subtle;
        visuals.widgets.noninteractive.fg_stroke =
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(208, 218, 234));
        visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(6);

        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(21, 29, 45);
        visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(18, 25, 39);
        visuals.widgets.inactive.bg_stroke = border_subtle;
        visuals.widgets.inactive.fg_stroke =
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(224, 232, 246));
        visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(6);

        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(30, 42, 65);
        visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(27, 38, 59);
        visuals.widgets.hovered.bg_stroke = border_hover;
        visuals.widgets.hovered.fg_stroke =
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(246, 251, 255));
        visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(6);

        visuals.widgets.active.bg_fill = accent_fill;
        visuals.widgets.active.weak_bg_fill = accent_fill;
        visuals.widgets.active.bg_stroke = accent_stroke;
        visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0_f32, egui::Color32::WHITE);
        visuals.widgets.active.corner_radius = egui::CornerRadius::same(6);

        visuals.selection.bg_fill = accent_fill;
        visuals.selection.stroke = accent_stroke;
        visuals.window_corner_radius = egui::CornerRadius::same(10);
        visuals.window_stroke = border_subtle;

        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(7.0, 5.5);
        style.spacing.button_padding = egui::vec2(8.0, 4.5);
        style.spacing.window_margin = egui::Margin::same(10);
        style.visuals = visuals;
        ctx.set_style(style);
    }

    fn dir_to_angles(dir: Vec3) -> (f32, f32) {
        let d = dir.normalize_or_zero();
        let el = d.y.clamp(-1.0, 1.0).asin().to_degrees();
        let az = d.x.atan2(d.z).to_degrees();
        (az, el)
    }

    pub fn sync_sun_angles_from_dir(&mut self) {
        let (az, el) = Self::dir_to_angles(self.mesh_config.light_dir);
        self.sun_azimuth_deg = az;
        self.sun_elevation_deg = el;
    }

    fn update_dir_from_sun_angles(&mut self) {
        let az = self.sun_azimuth_deg.to_radians();
        let el = self.sun_elevation_deg.to_radians();
        self.mesh_config.light_dir = Vec3::new(
            el.cos() * az.sin(),
            el.sin(),
            el.cos() * az.cos(),
        )
        .normalize_or_zero();
    }

    pub fn apply_fluid_preset(&mut self, preset: FluidPhysicsPreset) {
        self.fluid_preset = preset;
        preset.apply(&mut self.solver.config);
        self.status_message = format!("Fluid substance: {}", preset.name());
    }

    pub fn apply_matching_visual_material(&mut self) {
        let mat = match self.fluid_preset {
            FluidPhysicsPreset::Water => FluidMaterialPreset::ClearWater,
            FluidPhysicsPreset::Ocean => FluidMaterialPreset::DeepOcean,
            FluidPhysicsPreset::Oil | FluidPhysicsPreset::Honey | FluidPhysicsPreset::Lava => {
                FluidMaterialPreset::MoltenAmber
            }
            FluidPhysicsPreset::Glycerin => FluidMaterialPreset::EmeraldGel,
            FluidPhysicsPreset::Mercury => FluidMaterialPreset::LiquidChrome,
            FluidPhysicsPreset::ZeroG => FluidMaterialPreset::TropicalLagoon,
            FluidPhysicsPreset::Custom => self.mesh_config.preset,
        };
        mat.apply(&mut self.mesh_config);
        self.color_map = match self.fluid_preset {
            FluidPhysicsPreset::Lava => ColorMapType::Inferno,
            FluidPhysicsPreset::Honey | FluidPhysicsPreset::Oil => ColorMapType::Fire,
            FluidPhysicsPreset::Mercury => ColorMapType::Chrome,
            FluidPhysicsPreset::Glycerin => ColorMapType::Viridis,
            _ => ColorMapType::Ocean,
        };
        self.status_message = format!("Matched optical material: {}", mat.name());
    }

    pub fn set_2d(&mut self, is_2d: bool) {
        self.is_2d = is_2d;
        self.solver.set_2d(is_2d);
        if is_2d {
            self.camera.azimuth = 0.0;
            self.camera.elevation = 0.0;
            self.camera.target = Vec3::new(0.0, 0.6, 0.0);
            self.camera.distance = 2.4;
            self.camera_orbit_speed_deg = 0.0;
        }
        self.solver.apply_quality(self.current_quality);
        self.load_preset(self.current_preset);
        self.status_message = if is_2d {
            "Switched to 2D Planar Mode".to_string()
        } else {
            "Switched to 3D Simulation Mode".to_string()
        };
    }

    pub fn load_preset(&mut self, preset: ScenePreset) {
        self.current_preset = preset;
        SceneGenerator::setup_scene(&mut self.solver, preset);
        if preset == ScenePreset::ObstacleDrop {
            self.sphere_obstacle_enabled = true;
        }
        // Re-apply active non-default FluidPhysicsPreset (unless ZeroGravityBlobs explicitly overrides gravity)
        if preset == ScenePreset::ZeroGravityBlobs {
            self.fluid_preset = FluidPhysicsPreset::ZeroG;
        } else if self.fluid_preset != FluidPhysicsPreset::Water
            && self.fluid_preset != FluidPhysicsPreset::Custom
        {
            self.fluid_preset.apply(&mut self.solver.config);
        }
        if let Some(ref mut gpu) = self.gpu_solver {
            gpu.init_buffers(
                self.solver.bounds,
                &self.solver.config,
                &self.solver.particles,
            );
        }
        self.next_collider_id = self
            .solver
            .colliders
            .iter()
            .map(|c| c.id + 1)
            .max()
            .unwrap_or(1)
            .max(self.next_collider_id);
        self.sim_time = 0.0;
        self.mesh_dirty = true;
        self.status_message = format!("Loaded preset: {}", preset.name());
    }

    pub fn set_quality(&mut self, quality: SimulationQuality) {
        self.current_quality = quality;
        self.solver.apply_quality(quality);
        self.load_preset(self.current_preset);
        self.status_message = format!("Set quality: {}", quality.name_for(self.is_2d));
    }

    pub fn reset_camera_framing(&mut self) {
        if self.is_2d {
            self.camera.azimuth = 0.0;
            self.camera.elevation = 0.0;
            self.camera.target = Vec3::new(0.0, 0.6, 0.0);
            self.camera.distance = 2.4;
        } else {
            self.camera.reset();
        }
        self.camera_orbit_speed_deg = 0.0;
    }

    /// Snapshot complete studio state into a portable [`SceneConfigFile`].
    pub fn to_scene_config_file(&self) -> SceneConfigFile {
        SceneConfigFile {
            version: 1,
            preset: self.current_preset.slug().to_string(),
            quality: self.current_quality.slug().to_string(),
            fluid_preset: self.fluid_preset.slug().to_string(),
            is_2d: self.is_2d,
            sim_speed: self.sim_speed,
            mouse_strength: self.solver.mouse_force_strength,
            solver: SolverConfigDto::from_config(&self.solver.config),
            chamber: ChamberDto::from_chamber(&self.solver.chamber),
            colliders: self
                .solver
                .colliders
                .iter()
                .map(ColliderDto::from_collider)
                .collect(),
            faucets: self
                .solver
                .faucets
                .iter()
                .map(FaucetDto::from_faucet)
                .collect(),
            drains: self
                .solver
                .drains
                .iter()
                .map(DrainDto::from_drain)
                .collect(),
            camera: CameraDto::from_camera(&self.camera, self.camera_orbit_speed_deg),
            visuals: VisualsDto {
                render_mode: self.render_mode.slug().to_string(),
                color_mode: self.color_mode.slug().to_string(),
                color_map: self.color_map.slug().to_string(),
                particle_scale: self.display_size_factor,
                particle_foam: self.particle_foam,
                lit_spheres: self.lit_spheres,
                enable_culling: self.enable_culling,
                mesh: SurfaceMeshDto::from_config(&self.mesh_config),
            },
        }
    }

    /// Restore full studio state from a loaded [`SceneConfigFile`].
    pub fn apply_scene_config_file(&mut self, scene: &SceneConfigFile) {
        self.is_2d = scene.is_2d;
        self.solver.set_2d(scene.is_2d);
        self.current_preset = scene.parsed_preset();
        let mut q = scene.parsed_quality();
        if q.is_cli_only() {
            q = SimulationQuality::Extreme;
        }
        self.current_quality = q;
        self.fluid_preset = scene.parsed_fluid_preset();
        self.sim_speed = scene.sim_speed.clamp(0.05, 4.0);
        self.solver.mouse_force_strength = scene.mouse_strength.clamp(1.0, 200.0);

        let chamber = scene.chamber.to_chamber();
        self.solver.set_chamber(chamber);
        self.solver.config = scene.solver.to_config();
        self.solver.config.rest_density = if self.is_2d {
            compute_lattice_rest_density_2d(
                self.solver.config.smoothing_length,
                self.solver.config.spacing,
            )
        } else {
            compute_lattice_rest_density(
                self.solver.config.smoothing_length,
                self.solver.config.spacing,
            )
        };
        let saved_cfg = self.solver.config.clone();
        SceneGenerator::setup_scene(&mut self.solver, self.current_preset);
        self.solver.config = saved_cfg;

        // Replace scene objects with saved colliders, faucets, and drains
        self.solver.colliders.clear();
        for (i, c_dto) in scene.colliders.iter().enumerate() {
            if let Some(col) = c_dto.to_collider((i + 1) as u32) {
                self.solver.colliders.push(col);
            }
        }
        self.next_collider_id = (self.solver.colliders.len() as u32) + 1;

        self.solver.faucets.clear();
        for (i, f_dto) in scene.faucets.iter().enumerate() {
            self.solver.faucets.push(f_dto.to_faucet((i + 1) as u32));
        }
        self.next_faucet_id = (self.solver.faucets.len() as u32) + 1;

        self.solver.drains.clear();
        for (i, d_dto) in scene.drains.iter().enumerate() {
            self.solver.drains.push(d_dto.to_drain((i + 1) as u32));
        }
        self.next_drain_id = (self.solver.drains.len() as u32) + 1;

        self.camera = scene.camera.to_camera();
        self.camera_orbit_speed_deg = if self.is_2d {
            0.0
        } else {
            scene.camera.orbit_speed_deg.clamp(-180.0, 180.0)
        };

        self.render_mode = scene.visuals.parsed_render_mode();
        self.color_mode = scene.visuals.parsed_color_mode();
        self.color_map = scene.visuals.parsed_color_map();
        self.display_size_factor = scene.visuals.particle_scale.clamp(0.3, 3.0);
        self.particle_foam = scene.visuals.particle_foam;
        self.lit_spheres = scene.visuals.lit_spheres;
        self.enable_culling = scene.visuals.enable_culling;
        self.mesh_config = scene.visuals.mesh.to_config();
        self.sync_sun_angles_from_dir();

        if let Some(ref mut gpu) = self.gpu_solver {
            gpu.init_buffers(
                self.solver.bounds,
                &self.solver.config,
                &self.solver.particles,
            );
        }
        self.sim_time = 0.0;
        self.mesh_dirty = true;
    }

    pub fn save_scene_json(&mut self, path: &str) {
        let trimmed = path.trim();
        let target = if trimmed.is_empty() {
            "scene.json"
        } else {
            trimmed
        };
        match self.to_scene_config_file().save_to_path(target) {
            Ok(_) => {
                self.scene_file_path = target.to_string();
                self.status_message = format!("Saved studio scene to {}", target);
            }
            Err(e) => {
                self.status_message = e;
            }
        }
    }

    pub fn load_scene_json(&mut self, path: &str) {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            self.status_message = "Scene load error: empty file path".to_string();
            return;
        }
        match SceneConfigFile::load_from_path(trimmed) {
            Ok(scene) => {
                self.apply_scene_config_file(&scene);
                self.scene_file_path = trimmed.to_string();
                self.status_message = format!("Loaded studio scene from {}", trimmed);
            }
            Err(e) => {
                self.status_message = e;
            }
        }
    }

    fn step_simulation_once(&mut self, effective_dt: f32) {
        let start = Instant::now();

        if self.use_gpu && self.gpu_solver.is_some() {
            let prev_len = self.solver.particles.len();
            self.solver.emit_faucets(effective_dt);
            let emitted = self.solver.particles.len() != prev_len;

            if let Some(ref mut gpu) = self.gpu_solver {
                if emitted {
                    // Emitted particles are strictly appended at the tail of `solver.particles`
                    gpu.sync_particles(
                        self.solver.bounds,
                        &self.solver.config,
                        &self.solver.particles,
                        true,
                    );
                }
                gpu.step(
                    self.solver.bounds,
                    &self.solver.chamber,
                    &self.solver.config,
                    &self.solver.colliders,
                    self.solver.mouse_ray,
                    self.solver.mouse_force_strength,
                    self.solver.mouse_force_mode,
                    effective_dt,
                    self.is_2d,
                );
                gpu.read_particles(&mut self.solver.particles);
            }

            // Resolve any custom .obj Mesh colliders on readback
            let has_mesh_collider = self
                .solver
                .colliders
                .iter()
                .any(|c| c.enabled && matches!(c.shape, ColliderShape::Mesh(_)));
            if has_mesh_collider {
                let radius = self.solver.config.particle_radius;
                let colliders = &self.solver.colliders;
                let is_2d = self.is_2d;
                let b_min = self.solver.bounds.min + Vec3::splat(radius);
                let b_max = self.solver.bounds.max - Vec3::splat(radius);
                let any_mesh_hit = self
                    .solver
                    .particles
                    .par_iter_mut()
                    .map(|p| {
                        let mut p_hit = false;
                        for col in colliders {
                            if let ColliderShape::Mesh(_) = &col.shape {
                                p_hit |= col.resolve(&mut p.position, &mut p.velocity, radius);
                            }
                        }
                        if p_hit {
                            p.position = p.position.clamp(b_min.min(b_max), b_max.max(b_min));
                            if is_2d {
                                p.position.z = 0.0;
                                p.velocity.z = 0.0;
                            }
                        }
                        p_hit
                    })
                    .reduce(|| false, |a, b| a || b);
                if any_mesh_hit {
                    if let Some(ref mut gpu) = self.gpu_solver {
                        gpu.sync_particles(
                            self.solver.bounds,
                            &self.solver.config,
                            &self.solver.particles,
                            false,
                        );
                    }
                }
            }

            let drained = self.solver.drain_particles();
            if drained > 0 {
                if let Some(ref mut gpu) = self.gpu_solver {
                    // Draining compacts interior indices (`retain`), so full upload (`append_only = false`) is required
                    gpu.sync_particles(
                        self.solver.bounds,
                        &self.solver.config,
                        &self.solver.particles,
                        false,
                    );
                }
            }
        } else {
            self.solver.step(effective_dt);
        }

        self.sim_time += effective_dt;
        self.mesh_dirty = true;
        self.last_step_time_ms = start.elapsed().as_secs_f32() * 1000.0;
    }

    fn make_preview_frame(
        rgb: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> PreviewFrameRgb {
        if src_w == dst_w && src_h == dst_h {
            return PreviewFrameRgb {
                width: src_w,
                height: src_h,
                rgb: Arc::new(rgb.to_vec()),
            };
        }
        let mut out = vec![0u8; dst_w * dst_h * 3];
        let sx = src_w as f32 / dst_w.max(1) as f32;
        let sy = src_h as f32 / dst_h.max(1) as f32;
        for y in 0..dst_h {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (src_h.saturating_sub(1)) as f32);
            let y0 = fy as usize;
            let y1 = (y0 + 1).min(src_h.saturating_sub(1));
            let wy = fy - y0 as f32;
            for x in 0..dst_w {
                let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (src_w.saturating_sub(1)) as f32);
                let x0 = fx as usize;
                let x1 = (x0 + 1).min(src_w.saturating_sub(1));
                let wx = fx - x0 as f32;

                let i00 = (y0 * src_w + x0) * 3;
                let i10 = (y0 * src_w + x1) * 3;
                let i01 = (y1 * src_w + x0) * 3;
                let i11 = (y1 * src_w + x1) * 3;
                let dst = (y * dst_w + x) * 3;
                for c in 0..3 {
                    let top = rgb[i00 + c] as f32 * (1.0 - wx) + rgb[i10 + c] as f32 * wx;
                    let bot = rgb[i01 + c] as f32 * (1.0 - wx) + rgb[i11 + c] as f32 * wx;
                    out[dst + c] = (top * (1.0 - wy) + bot * wy + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
        PreviewFrameRgb {
            width: dst_w,
            height: dst_h,
            rgb: Arc::new(out),
        }
    }

    pub fn export_current_png(&mut self) {
        let dir = std::path::Path::new(&self.export_path);
        if let Err(e) = std::fs::create_dir_all(dir) {
            self.status_message = format!("PNG export error: {}", e);
            return;
        }
        let path = dir.join(format!("fluid_frame_{:05}.png", self.exported_frames_count));
        let width = self.video_width.max(320);
        let height = self.video_height.max(240);
        let t_start = Instant::now();

        let mut mcfg = self.mesh_config.clone();
        self.gui_render_quality
            .apply_to_mesh_config(&mut mcfg, false, false);

        let mut off = GuiOfflineRenderer::new(
            width,
            height,
            self.gui_render_quality,
            self.use_gpu,
            None,
            self.enable_culling,
            self.lit_spheres,
            self.particle_foam,
            &mcfg,
        );
        let backend_lbl = off.backend_label();
        let rgb_slice = off.render_frame(
            self.render_mode,
            &self.solver.particles,
            None,
            self.solver.config.particle_radius,
            self.solver.config.smoothing_length,
            self.display_size_factor,
            self.solver.config.rest_density,
            &mcfg,
            &self.camera,
            &self.solver.chamber,
            self.solver.bounds,
            &self.solver.colliders,
            self.color_mode,
            self.color_map,
            self.sim_time,
            self.is_2d,
        );

        let rgb_vec = rgb_slice.to_vec();
        let elapsed = t_start.elapsed().as_secs_f32();

        match PngExporter::save_rgb(&path, width, height, &rgb_vec) {
            Ok(_) => {
                let path_str = path.display().to_string();
                self.status_message = format!("Saved {}x{} PNG to {}", width, height, path_str);
                self.exported_frames_count += 1;
                if let Ok(mut g) = self.bg_render.lock() {
                    if !g.is_running {
                        g.kind = StudioRenderKind::StillFrame;
                        g.completed_frames = 1;
                        g.total_frames = 1;
                        g.elapsed_secs = elapsed;
                        g.backend_label = backend_lbl;
                        g.output_width = width;
                        g.output_height = height;
                        g.quality = self.gui_render_quality;
                        g.saved_path = Some(path_str);
                        g.full_res_frame = Some(PreviewFrameRgb {
                            width,
                            height,
                            rgb: Arc::new(rgb_vec),
                        });
                        g.anim_frames.clear();
                        g.frame_revision = g.frame_revision.wrapping_add(1);
                    }
                }
            }
            Err(e) => {
                self.status_message = format!("PNG export error: {}", e);
            }
        }
    }

    pub fn effective_render_mode(&self) -> RenderMode {
        match self.render_appearance {
            RenderAppearance::MatchViewport => self.render_mode,
            RenderAppearance::SmoothLiquid => RenderMode::SurfaceMesh,
            RenderAppearance::ParticleSpheres => RenderMode::Particles,
        }
    }

    pub fn apply_guided_render_preset(&mut self, preset: GuidedRenderPreset) {
        self.guided_render_preset = preset;
        match preset {
            GuidedRenderPreset::QuickPreview => {
                self.render_kind = StudioRenderKind::Animation;
                self.render_appearance = RenderAppearance::SmoothLiquid;
                self.render_sim_quality = Some(SimulationQuality::High);
                self.gui_render_quality = CliRenderQuality::Standard;
                self.video_width = 1280;
                self.video_height = 720;
                self.cli_fps = 30;
                self.cli_duration = 4.0;
            }
            GuidedRenderPreset::RecommendedHd => {
                self.render_kind = StudioRenderKind::Animation;
                self.render_appearance = RenderAppearance::SmoothLiquid;
                self.render_sim_quality = Some(SimulationQuality::Ultra);
                self.gui_render_quality = CliRenderQuality::Studio;
                self.video_width = 1920;
                self.video_height = 1080;
                self.cli_fps = 60;
                self.cli_duration = 5.0;
            }
            GuidedRenderPreset::Cinematic2K => {
                self.render_kind = StudioRenderKind::Animation;
                self.render_appearance = RenderAppearance::SmoothLiquid;
                self.render_sim_quality = Some(SimulationQuality::Cinematic);
                self.gui_render_quality = CliRenderQuality::Cinema;
                self.video_width = 2560;
                self.video_height = 1440;
                self.cli_fps = 60;
                self.cli_duration = 5.0;
            }
            GuidedRenderPreset::Ultra4K => {
                self.render_kind = StudioRenderKind::Animation;
                self.render_appearance = RenderAppearance::SmoothLiquid;
                self.render_sim_quality = Some(SimulationQuality::Extreme);
                self.gui_render_quality = CliRenderQuality::Cinema;
                self.video_width = 3840;
                self.video_height = 2160;
                self.cli_fps = 60;
                self.cli_duration = 5.0;
            }
            GuidedRenderPreset::StudioPhoto => {
                self.render_kind = StudioRenderKind::StillFrame;
                self.render_appearance = RenderAppearance::SmoothLiquid;
                self.render_sim_quality = Some(SimulationQuality::Cinematic);
                self.gui_render_quality = CliRenderQuality::Cinema;
                self.video_width = 2560;
                self.video_height = 1440;
                if self.sim_time < 0.05 && self.current_preset != ScenePreset::Empty {
                    self.render_still_simulate = true;
                    self.render_still_time = 1.6;
                } else {
                    self.render_still_simulate = false;
                    self.render_still_time = self.sim_time;
                }
                self.render_auto_save_png = true;
            }
            GuidedRenderPreset::Custom => {}
        }
    }

    fn format_particle_count(n: usize) -> String {
        let s = n.to_string();
        let mut out = String::with_capacity(s.len() + s.len() / 3);
        for (i, ch) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                out.push(',');
            }
            out.push(ch);
        }
        out.chars().rev().collect()
    }

    fn open_in_file_manager(path_str: &str) {
        let p = std::path::Path::new(path_str.trim());
        let dir = if p.is_dir() {
            p.to_path_buf()
        } else if let Some(parent) = p.parent() {
            if parent.as_os_str().is_empty() {
                std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
            } else {
                parent.to_path_buf()
            }
        } else {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
        };

        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg(&dir).spawn();
        #[cfg(target_os = "windows")]
        let _ = std::process::Command::new("explorer").arg(&dir).spawn();
    }

    pub fn generate_render_tab_cli_command(&self) -> String {
        let mut clone_self_mode = self.cli_mode;
        let _ = &mut clone_self_mode;
        // Build an exact CLI command string reflecting the Render tab's offline quality & appearance
        let mut parts = vec!["liquid-rust".to_string()];
        let eff_mode = self.effective_render_mode();
        let eff_qual = self.render_sim_quality.unwrap_or(self.current_quality);

        match self.render_kind {
            StudioRenderKind::Animation => {
                let out = if self.cli_mp4_output.trim().is_empty() {
                    "simulation.mp4"
                } else {
                    self.cli_mp4_output.trim()
                };
                parts.push(format!("--export-mp4 {}", out));
                parts.push(format!("--time {:.1}", self.cli_duration));
                parts.push(format!("--fps {}", self.cli_fps));
            }
            StudioRenderKind::StillFrame => {
                let out = if self.png_output_path.trim().is_empty() {
                    "snapshot.png"
                } else {
                    self.png_output_path.trim()
                };
                parts.push(format!("--export-png {}", out));
                let t = if self.render_still_simulate {
                    self.render_still_time
                } else {
                    self.sim_time
                };
                if t > 0.001 {
                    parts.push(format!("--time {:.2}", t));
                    parts.push(format!("--fps {}", self.cli_fps));
                } else {
                    parts.push("--frames 0".to_string());
                }
            }
        }

        parts.push(format!(
            "--video-width {} --video-height {}",
            self.video_width, self.video_height
        ));
        if self.gui_render_quality != CliRenderQuality::Standard {
            parts.push(format!("--render-quality {}", self.gui_render_quality.slug()));
        }
        if self.is_2d {
            parts.push("--2d".to_string());
        }
        parts.push(format!("--preset {}", self.current_preset.slug()));
        if eff_qual != SimulationQuality::Custom {
            parts.push(format!("--quality {}", eff_qual.slug()));
        }
        if self.fluid_preset != FluidPhysicsPreset::Water
            && self.fluid_preset != FluidPhysicsPreset::Custom
        {
            parts.push(format!("--fluid-preset {}", self.fluid_preset.slug()));
        }
        if eff_mode == RenderMode::SurfaceMesh {
            parts.push("--surface-mesh".to_string());
            if self.mesh_config.preset != FluidMaterialPreset::ClearWater {
                parts.push(format!("--mesh-material {}", self.mesh_config.preset.slug()));
            }
        }
        if self.mesh_config.env_preset != EnvironmentPreset::StudioDark {
            parts.push(format!("--env-preset {}", self.mesh_config.env_preset.slug()));
        }
        if (self.sim_speed - 1.0).abs() > 1e-3 {
            parts.push(format!("--sim-speed {:.2}", self.sim_speed));
        }
        if !self.is_2d && self.camera_orbit_speed_deg.abs() > 0.05 {
            parts.push(format!("--camera-orbit {:.1}", self.camera_orbit_speed_deg));
        }
        if self.render_also_export_obj {
            parts.push(format!("--export-obj {}", self.export_path.trim()));
        }
        if self.render_also_export_ply {
            parts.push(format!("--export-ply {}", self.export_path.trim()));
        }
        if !self.use_gpu {
            parts.push("--cpu".to_string());
        }
        parts.join(" ")
    }

    pub fn save_rendered_result_png_to_path(&mut self, path: &std::path::Path) {
        let frame_opt = {
            if let Ok(g) = self.bg_render.lock() {
                if !g.anim_frames.is_empty() {
                    let idx = self.anim_play_idx.min(g.anim_frames.len() - 1);
                    if idx + 1 == g.anim_frames.len() && g.full_res_frame.is_some() {
                        g.full_res_frame.clone()
                    } else {
                        Some(g.anim_frames[idx].clone())
                    }
                } else {
                    g.full_res_frame.clone()
                }
            } else {
                None
            }
        };

        let Some(frame) = frame_opt else {
            self.status_message = "No rendered image available to save yet".to_string();
            return;
        };

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    self.status_message = format!("PNG save error: {}", e);
                    return;
                }
            }
        }
        match PngExporter::save_rgb(path, frame.width, frame.height, &frame.rgb) {
            Ok(_) => {
                let p_str = path.display().to_string();
                self.exported_frames_count += 1;
                self.status_message = format!(
                    "Saved rendered image ({}x{}) to {}",
                    frame.width, frame.height, p_str
                );
                if let Ok(mut g) = self.bg_render.lock() {
                    g.saved_path = Some(p_str);
                }
            }
            Err(e) => {
                self.status_message = format!("PNG save error: {}", e);
            }
        }
    }

    /// Save the currently displayed studio render result (full-res still or current animation frame) to PNG.
    pub fn save_rendered_result_png(&mut self) {
        let dir = std::path::Path::new(&self.export_path);
        let path = dir.join(format!("render_result_{:05}.png", self.exported_frames_count));
        self.save_rendered_result_png_to_path(&path);
    }

    #[allow(clippy::too_many_arguments)]
    fn step_offline_solver_frame(
        render_solver: &mut SphSolver,
        gpu_opt: &mut Option<GpuSphSolver>,
        cpu_particles_synced: &mut bool,
        need_cpu_particles: bool,
        has_mesh_col: bool,
        dt: f32,
        is_2d: bool,
    ) {
        if let Some(ref mut gpu) = gpu_opt {
            let prev_n = render_solver.particles.len();
            render_solver.emit_faucets(dt);
            let new_n = render_solver.particles.len();
            if new_n != prev_n {
                if new_n > gpu.particle_capacity && !*cpu_particles_synced {
                    let newly_emitted = render_solver.particles.split_off(prev_n);
                    gpu.read_particles(&mut render_solver.particles);
                    render_solver.particles.extend(newly_emitted);
                }
                gpu.sync_particles(
                    render_solver.bounds,
                    &render_solver.config,
                    &render_solver.particles,
                    true,
                );
            }
            gpu.step(
                render_solver.bounds,
                &render_solver.chamber,
                &render_solver.config,
                &render_solver.colliders,
                None,
                0.0,
                0,
                dt,
                is_2d,
            );

            if need_cpu_particles {
                gpu.read_particles(&mut render_solver.particles);
                *cpu_particles_synced = true;

                if has_mesh_col {
                    let radius = render_solver.config.particle_radius;
                    let colliders = &render_solver.colliders;
                    let b_min = render_solver.bounds.min + Vec3::splat(radius);
                    let b_max = render_solver.bounds.max - Vec3::splat(radius);
                    let hit = render_solver
                        .particles
                        .par_iter_mut()
                        .map(|p| {
                            let mut p_hit = false;
                            for col in colliders {
                                if let ColliderShape::Mesh(_) = &col.shape {
                                    p_hit |= col.resolve(&mut p.position, &mut p.velocity, radius);
                                }
                            }
                            if p_hit {
                                p.position = p.position.clamp(b_min.min(b_max), b_max.max(b_min));
                                if is_2d {
                                    p.position.z = 0.0;
                                    p.velocity.z = 0.0;
                                }
                            }
                            p_hit
                        })
                        .reduce(|| false, |a, b| a || b);
                    if hit {
                        gpu.sync_particles(
                            render_solver.bounds,
                            &render_solver.config,
                            &render_solver.particles,
                            false,
                        );
                    }
                }

                let drained = render_solver.drain_particles();
                if drained > 0 {
                    gpu.sync_particles(
                        render_solver.bounds,
                        &render_solver.config,
                        &render_solver.particles,
                        false,
                    );
                }
            } else {
                *cpu_particles_synced = false;
            }
        } else {
            render_solver.step(dt);
        }
    }

    /// Launch a non-blocking Studio Still Frame render (either instant from the live editor state,
    /// or simulated offline at CLI particle density up to the chosen capture timestamp) and display the end result.
    pub fn start_gui_still_render(&mut self, ctx: Option<egui::Context>, save_to_disk: bool) {
        if let Ok(guard) = self.bg_render.lock() {
            if guard.is_running {
                return;
            }
        }

        let width = self.video_width.max(320);
        let height = self.video_height.max(240);
        let rq = self.gui_render_quality;
        let render_mode = self.effective_render_mode();
        let mut mesh_cfg = self.mesh_config.clone();
        rq.apply_to_mesh_config(&mut mesh_cfg, false, false);

        let override_q = self
            .render_sim_quality
            .filter(|&q| q != self.current_quality && q != SimulationQuality::Custom);
        let need_fresh_sim = self.render_still_simulate || override_q.is_some();
        let target_t = if self.render_still_simulate {
            self.render_still_time.max(0.0)
        } else {
            self.sim_time.max(0.0)
        };
        let fps = self.cli_fps.max(1);
        let sim_steps = if need_fresh_sim && target_t > 1e-4 {
            (target_t * fps as f32).round().max(1.0) as usize
        } else {
            0
        };
        let dt = (1.0 / fps as f32) * self.sim_speed;

        let mut desired_cfg = self.solver.config.clone();
        if let Some(q) = self.render_sim_quality {
            q.apply(&mut desired_cfg, self.is_2d);
            desired_cfg.substeps = desired_cfg.substeps.max(self.solver.config.substeps);
            desired_cfg.solver_iterations = desired_cfg
                .solver_iterations
                .max(self.solver.config.solver_iterations);
        }

        let live_particles_opt = if need_fresh_sim {
            None
        } else {
            Some(self.solver.particles.clone())
        };
        let current_preset = self.current_preset;
        let display_size_factor = self.display_size_factor;
        let mut camera = self.camera.clone();
        let orbit_rad_per_sec = if self.is_2d {
            0.0
        } else {
            self.camera_orbit_speed_deg.to_radians()
        };
        let wall_frame_dt = 1.0 / fps as f32;
        let chamber = self.solver.chamber.clone();
        let bounds = self.solver.bounds;
        let colliders = self.solver.colliders.clone();
        let faucets = self.solver.faucets.clone();
        let drains = self.solver.drains.clone();
        let color_mode = self.color_mode;
        let color_map = self.color_map;
        let initial_sim_time = if need_fresh_sim { 0.0 } else { self.sim_time };
        let is_2d = self.is_2d;
        let use_gpu = self.use_gpu;
        let enable_culling = self.enable_culling;
        let lit_spheres = self.lit_spheres;
        let particle_foam = self.particle_foam;

        let save_target: Option<std::path::PathBuf> = if save_to_disk {
            let trimmed = self.png_output_path.trim();
            if !trimmed.is_empty() && trimmed != "snapshot.png" {
                Some(std::path::PathBuf::from(trimmed))
            } else {
                let dir = std::path::PathBuf::from(&self.export_path);
                let idx = self.exported_frames_count;
                self.exported_frames_count += 1;
                Some(dir.join(format!("render_photo_{:05}.png", idx)))
            }
        } else {
            None
        };

        self.is_playing = false;
        self.anim_playing = false;
        self.bg_render_cancel.store(false, Ordering::SeqCst);
        let cancel_flag = Arc::clone(&self.bg_render_cancel);
        let state_arc = Arc::clone(&self.bg_render);

        if let Ok(mut g) = state_arc.lock() {
            g.is_running = true;
            g.kind = StudioRenderKind::StillFrame;
            g.completed_frames = 0;
            g.total_frames = sim_steps + 1;
            g.elapsed_secs = 0.0;
            g.phase_label = if need_fresh_sim {
                "Spawning fluid particles...".to_string()
            } else {
                "Rendering high-resolution photo...".to_string()
            };
            g.sim_particles_count = live_particles_opt.as_ref().map_or(0, |v| v.len());
            g.output_width = width;
            g.output_height = height;
            g.quality = rq;
            g.saved_path = None;
            g.finished_message = None;
            g.just_finished_anim = false;
        }
        self.status_message = format!(
            "Rendering {}x{} Studio photo ({})...",
            width,
            height,
            rq.name()
        );

        std::thread::spawn(move || {
            let t_start = Instant::now();
            let mut render_solver = SphSolver::new(bounds, desired_cfg.clone());
            render_solver.set_2d(is_2d);
            render_solver.set_chamber(chamber);
            if let Some(live_pts) = live_particles_opt {
                render_solver.particles = live_pts;
            } else {
                SceneGenerator::setup_scene(&mut render_solver, current_preset);
            }
            render_solver.config = desired_cfg;
            render_solver.colliders = colliders;
            render_solver.faucets = faucets;
            render_solver.drains = drains;

            let p_count = render_solver.particles.len();
            if let Ok(mut g) = state_arc.lock() {
                g.sim_particles_count = p_count;
            }

            let mut gpu_opt = if use_gpu && sim_steps > 0 {
                GpuSphSolver::new(
                    render_solver.bounds,
                    &render_solver.config,
                    &render_solver.particles,
                )
                .ok()
            } else {
                None
            };

            let mut anim_t = initial_sim_time;
            if sim_steps > 0 {
                let has_mesh_col = render_solver
                    .colliders
                    .iter()
                    .any(|c| c.enabled && matches!(c.shape, ColliderShape::Mesh(_)));
                let has_drains = !render_solver.drains.is_empty();
                let mut cpu_particles_synced = true;

                for step in 0..sim_steps {
                    if cancel_flag.load(Ordering::SeqCst) {
                        break;
                    }
                    let is_last = step + 1 == sim_steps;
                    let need_cpu = has_mesh_col
                        || has_drains
                        || (is_last && render_mode == RenderMode::SurfaceMesh);
                    Self::step_offline_solver_frame(
                        &mut render_solver,
                        &mut gpu_opt,
                        &mut cpu_particles_synced,
                        need_cpu,
                        has_mesh_col,
                        dt,
                        is_2d,
                    );
                    anim_t += dt;
                    camera.azimuth += orbit_rad_per_sec * wall_frame_dt;

                    if let Ok(mut g) = state_arc.lock() {
                        g.completed_frames = step + 1;
                        g.elapsed_secs = t_start.elapsed().as_secs_f32();
                        g.sim_particles_count = gpu_opt
                            .as_ref()
                            .map_or(render_solver.particles.len(), |gs| gs.particle_count);
                        g.phase_label = format!(
                            "Simulating fluid step {} / {} ({:.2}s)",
                            step + 1,
                            sim_steps,
                            anim_t
                        );
                    }
                    if let Some(ref c) = ctx {
                        c.request_repaint();
                    }
                }
                if let Some(ref gpu) = gpu_opt {
                    if render_mode == RenderMode::SurfaceMesh || !gpu.supports_vertex_storage() {
                        gpu.read_particles(&mut render_solver.particles);
                    }
                }
            }

            if let Ok(mut g) = state_arc.lock() {
                g.phase_label = format!("Painting {}×{} studio image...", width, height);
            }
            if let Some(ref c) = ctx {
                c.request_repaint();
            }

            let mut off = GuiOfflineRenderer::new(
                width,
                height,
                rq,
                use_gpu,
                gpu_opt.as_ref(),
                enable_culling,
                lit_spheres,
                particle_foam,
                &mesh_cfg,
            );
            if matches!(off, GuiOfflineRenderer::Cpu(_)) {
                if let Some(ref gpu) = gpu_opt {
                    gpu.read_particles(&mut render_solver.particles);
                }
            }
            let backend_lbl = off.backend_label();
            let rgb_slice = off.render_frame(
                render_mode,
                &render_solver.particles,
                gpu_opt.as_ref(),
                render_solver.config.particle_radius,
                render_solver.config.smoothing_length,
                display_size_factor,
                render_solver.config.rest_density,
                &mesh_cfg,
                &camera,
                &render_solver.chamber,
                render_solver.bounds,
                &render_solver.colliders,
                color_mode,
                color_map,
                anim_t,
                is_2d,
            );
            let rgb_vec = rgb_slice.to_vec();
            let elapsed = t_start.elapsed().as_secs_f32();

            let mut saved_path_str = None;
            let mut err_str = None;
            if let Some(full_p) = save_target {
                if let Some(parent) = full_p.parent() {
                    if !parent.as_os_str().is_empty() {
                        if let Err(e) = std::fs::create_dir_all(parent) {
                            err_str = Some(format!("PNG save error: {}", e));
                        }
                    }
                }
                if err_str.is_none() {
                    match PngExporter::save_rgb(&full_p, width, height, &rgb_vec) {
                        Ok(_) => saved_path_str = Some(full_p.display().to_string()),
                        Err(e) => err_str = Some(format!("PNG save error: {}", e)),
                    }
                }
            }

            let final_pts = gpu_opt
                .as_ref()
                .map_or(render_solver.particles.len(), |gs| gs.particle_count);

            if let Ok(mut g) = state_arc.lock() {
                g.is_running = false;
                g.kind = StudioRenderKind::StillFrame;
                g.completed_frames = sim_steps + 1;
                g.total_frames = sim_steps + 1;
                g.elapsed_secs = elapsed;
                g.phase_label = "Render complete".to_string();
                g.sim_particles_count = final_pts;
                g.backend_label = backend_lbl.clone();
                g.output_width = width;
                g.output_height = height;
                g.quality = rq;
                g.saved_path = saved_path_str.clone();
                g.full_res_frame = Some(PreviewFrameRgb {
                    width,
                    height,
                    rgb: Arc::new(rgb_vec),
                });
                g.anim_frames.clear();
                g.frame_revision = g.frame_revision.wrapping_add(1);
                g.finished_message = Some(if let Some(e) = err_str {
                    e
                } else if let Some(ref p) = saved_path_str {
                    format!(
                        "Rendered {}x{} photo in {:.2}s [{}] -> {}",
                        width, height, elapsed, backend_lbl, p
                    )
                } else {
                    format!(
                        "Rendered {}x{} studio photo in {:.2}s [{}]",
                        width, height, elapsed, backend_lbl
                    )
                });
            }
            if let Some(c) = ctx {
                c.request_repaint();
            }
        });
    }

    pub fn export_current_frame(&mut self) {
        let dir = std::path::Path::new(&self.export_path);
        if let Err(e) = std::fs::create_dir_all(dir) {
            self.status_message = format!("Export error: {}", e);
            return;
        }

        let filename = format!("fluid_frame_{:05}.ply", self.exported_frames_count);
        let path = dir.join(filename);

        let colors: Vec<[u8; 3]> = self
            .solver
            .particles
            .iter()
            .map(|p| self.compute_particle_color(p))
            .collect();

        match PlyExporter::save_binary(&path, &self.solver.particles, &colors) {
            Ok(_) => {
                self.status_message = format!("Exported PLY to {}", path.display());
                self.exported_frames_count += 1;
            }
            Err(e) => {
                self.status_message = format!("Export error: {}", e);
            }
        }
    }

    pub fn export_current_surface_obj(&mut self) {
        let dir = std::path::Path::new(&self.export_path);
        if let Err(e) = std::fs::create_dir_all(dir) {
            self.status_message = format!("OBJ export error: {}", e);
            return;
        }
        let mesh = SurfaceMesh::from_particles(
            &self.solver.particles,
            self.solver.config.particle_radius,
            self.solver.config.smoothing_length,
            &self.mesh_config,
        );
        let path = dir.join(format!("fluid_surface_{:05}.obj", self.exported_frames_count));
        match mesh.save_obj(&path) {
            Ok(_) => {
                self.status_message = format!(
                    "Saved surface mesh ({} tris) to {}",
                    mesh.triangle_count(),
                    path.display()
                );
                self.exported_frames_count += 1;
            }
            Err(e) => {
                self.status_message = format!("OBJ export error: {}", e);
            }
        }
    }

    /// Launch a non-blocking studio animation render directly from the GUI using current scene & optics settings.
    pub fn start_gui_mp4_render(&mut self) {
        self.start_gui_mp4_render_with_ctx(None);
    }

    /// Launch a non-blocking CLI-grade studio animation render, streaming rendered frames into the GUI Render Result player
    /// and optionally encoding an H.264 MP4 video + OBJ/PLY sequences.
    pub fn start_gui_mp4_render_with_ctx(&mut self, ctx: Option<egui::Context>) {
        if let Ok(guard) = self.bg_render.lock() {
            if guard.is_running {
                return;
            }
        }

        self.is_playing = false;
        self.anim_playing = false;

        let save_mp4 = self.render_save_mp4;
        let out_path = if self.cli_mp4_output.trim().is_empty() {
            "simulation.mp4".to_string()
        } else {
            self.cli_mp4_output.trim().to_string()
        };

        let fps = self.cli_fps.max(1);
        let total_frames = (self.cli_duration * fps as f32).round().max(1.0) as usize;
        let dt = (1.0 / fps as f32) * self.sim_speed;
        let width = ((self.video_width.max(320) + 1) & !1).max(320);
        let height = ((self.video_height.max(240) + 1) & !1).max(240);

        let max_budget_pixels =
            ((384usize * 1024 * 1024) / 3 / total_frames.max(1)).clamp(640 * 360, 1920 * 1080);
        let full_pixels = width * height;
        let (prev_w, prev_h) = if full_pixels <= max_budget_pixels {
            (width, height)
        } else {
            let scale = (max_budget_pixels as f64 / full_pixels as f64).sqrt();
            let pw = ((width as f64 * scale).round() as usize).max(160);
            let ph = ((height as f64 * scale).round() as usize).max(90);
            (pw, ph)
        };

        let override_q = self
            .render_sim_quality
            .filter(|&q| q != self.current_quality && q != SimulationQuality::Custom);
        let mut desired_cfg = self.solver.config.clone();
        if let Some(q) = self.render_sim_quality {
            q.apply(&mut desired_cfg, self.is_2d);
            desired_cfg.substeps = desired_cfg.substeps.max(self.solver.config.substeps);
            desired_cfg.solver_iterations = desired_cfg
                .solver_iterations
                .max(self.solver.config.solver_iterations);
        }

        let use_live_state =
            override_q.is_none() && self.render_anim_start == AnimationStartSource::CurrentState;
        let live_particles_opt = if use_live_state {
            Some(self.solver.particles.clone())
        } else {
            None
        };
        let initial_anim_t = if use_live_state { self.sim_time } else { 0.0 };

        let current_preset = self.current_preset;
        let bounds = self.solver.bounds;
        let chamber = self.solver.chamber.clone();
        let colliders = self.solver.colliders.clone();
        let faucets = self.solver.faucets.clone();
        let drains = self.solver.drains.clone();

        let render_mode = self.effective_render_mode();
        let rq = self.gui_render_quality;
        let mut mesh_cfg = self.mesh_config.clone();
        rq.apply_to_mesh_config(&mut mesh_cfg, false, false);
        let mut camera = self.camera.clone();
        let orbit_rad_per_sec = if self.is_2d {
            0.0
        } else {
            self.camera_orbit_speed_deg.to_radians()
        };
        let wall_frame_dt = 1.0 / fps as f32;
        let color_mode = self.color_mode;
        let color_map = self.color_map;
        let display_size_factor = self.display_size_factor;
        let enable_culling = self.enable_culling;
        let lit_spheres = self.lit_spheres;
        let particle_foam = self.particle_foam;
        let is_2d = self.is_2d;
        let use_gpu = self.use_gpu;
        let also_obj = self.render_also_export_obj;
        let also_ply = self.render_also_export_ply;
        let seq_dir = std::path::PathBuf::from(self.export_path.trim());

        self.bg_render_cancel.store(false, Ordering::SeqCst);
        let cancel_flag = Arc::clone(&self.bg_render_cancel);
        let state_arc = Arc::clone(&self.bg_render);

        if let Ok(mut guard) = state_arc.lock() {
            guard.is_running = true;
            guard.kind = StudioRenderKind::Animation;
            guard.completed_frames = 0;
            guard.total_frames = total_frames;
            guard.elapsed_secs = 0.0;
            guard.phase_label = "Initializing fluid particles & GPU buffers...".to_string();
            guard.sim_particles_count = live_particles_opt.as_ref().map_or(0, |v| v.len());
            guard.output_width = width;
            guard.output_height = height;
            guard.quality = rq;
            guard.fps = fps;
            guard.saved_path = None;
            guard.finished_message = None;
            guard.full_res_frame = None;
            guard.anim_frames.clear();
            guard.anim_frames.reserve(total_frames.min(1800));
            guard.anim_generation = guard.anim_generation.wrapping_add(1);
            guard.just_finished_anim = false;
        }
        self.status_message = if save_mp4 {
            format!(
                "Rendering {}x{} video ({} frames) -> {}...",
                width, height, total_frames, out_path
            )
        } else {
            format!(
                "Rendering {}x{} studio video ({} frames)...",
                width, height, total_frames
            )
        };

        std::thread::spawn(move || {
            let t_start = Instant::now();

            let mut render_solver = SphSolver::new(bounds, desired_cfg.clone());
            render_solver.set_2d(is_2d);
            render_solver.set_chamber(chamber);
            if let Some(live_pts) = live_particles_opt {
                render_solver.particles = live_pts;
            } else {
                SceneGenerator::setup_scene(&mut render_solver, current_preset);
            }
            render_solver.config = desired_cfg;
            render_solver.colliders = colliders;
            render_solver.faucets = faucets;
            render_solver.drains = drains;

            if let Ok(mut g) = state_arc.lock() {
                g.sim_particles_count = render_solver.particles.len();
            }

            if also_obj || also_ply {
                let _ = std::fs::create_dir_all(&seq_dir);
            }
            if save_mp4 {
                if let Some(parent) = std::path::Path::new(&out_path).parent() {
                    if !parent.as_os_str().is_empty() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                }
            }

            let mut mp4_warn: Option<String> = None;
            let mut encoder = if save_mp4 {
                match Mp4Encoder::new_with_encoding(
                    &out_path,
                    width,
                    height,
                    fps,
                    rq.ffmpeg_preset(),
                    rq.ffmpeg_crf(),
                ) {
                    Ok(enc) => Some(enc),
                    Err(e) => {
                        mp4_warn = Some(format!("MP4 file skipped ({})", e));
                        None
                    }
                }
            } else {
                None
            };

            let mut gpu_opt = if use_gpu {
                GpuSphSolver::new(
                    render_solver.bounds,
                    &render_solver.config,
                    &render_solver.particles,
                )
                .ok()
            } else {
                None
            };

            let mut off = GuiOfflineRenderer::new(
                width,
                height,
                rq,
                use_gpu,
                gpu_opt.as_ref(),
                enable_culling,
                lit_spheres,
                particle_foam,
                &mesh_cfg,
            );
            let backend_lbl = off.backend_label();
            if let Ok(mut g) = state_arc.lock() {
                g.backend_label = backend_lbl.clone();
            }

            let has_mesh_col = render_solver
                .colliders
                .iter()
                .any(|c| c.enabled && matches!(c.shape, ColliderShape::Mesh(_)));
            let has_drains = !render_solver.drains.is_empty();
            let need_mesh = render_mode == RenderMode::SurfaceMesh || also_obj;
            let need_cpu_render = matches!(off, GuiOfflineRenderer::Cpu(_))
                || gpu_opt
                    .as_ref()
                    .map_or(true, |gs| !gs.supports_vertex_storage());
            let need_cpu_particles =
                has_mesh_col || has_drains || need_mesh || need_cpu_render || also_ply;

            let mut anim_t = initial_anim_t;
            let mut done = 0usize;
            let mut cpu_particles_synced = true;

            for frame_idx in 0..total_frames {
                if cancel_flag.load(Ordering::SeqCst) {
                    break;
                }

                Self::step_offline_solver_frame(
                    &mut render_solver,
                    &mut gpu_opt,
                    &mut cpu_particles_synced,
                    need_cpu_particles,
                    has_mesh_col,
                    dt,
                    is_2d,
                );
                anim_t += dt;

                let rgb_slice = off.render_frame(
                    render_mode,
                    &render_solver.particles,
                    gpu_opt.as_ref(),
                    render_solver.config.particle_radius,
                    render_solver.config.smoothing_length,
                    display_size_factor,
                    render_solver.config.rest_density,
                    &mesh_cfg,
                    &camera,
                    &render_solver.chamber,
                    render_solver.bounds,
                    &render_solver.colliders,
                    color_mode,
                    color_map,
                    anim_t,
                    is_2d,
                );

                camera.azimuth += orbit_rad_per_sec * wall_frame_dt;

                if also_obj {
                    let obj_mesh = SurfaceMesh::from_particles(
                        &render_solver.particles,
                        render_solver.config.particle_radius,
                        render_solver.config.smoothing_length,
                        &mesh_cfg,
                    );
                    let _ =
                        obj_mesh.save_obj(seq_dir.join(format!("fluid_surface_{:05}.obj", frame_idx)));
                }
                if also_ply {
                    let rest_rho = render_solver.config.rest_density;
                    let bnds = render_solver.bounds;
                    let colors: Vec<[u8; 3]> = render_solver
                        .particles
                        .iter()
                        .map(|p| {
                            ColorMap::particle_color(
                                p,
                                color_mode,
                                color_map,
                                rest_rho,
                                &bnds,
                                particle_foam,
                            )
                        })
                        .collect();
                    let _ = PlyExporter::save_binary(
                        seq_dir.join(format!("fluid_{:05}.ply", frame_idx)),
                        &render_solver.particles,
                        &colors,
                    );
                }

                let preview = Self::make_preview_frame(rgb_slice, width, height, prev_w, prev_h);

                if let Some(ref mut enc) = encoder {
                    if let Err(e) = enc.write_frame(rgb_slice) {
                        mp4_warn = Some(format!("MP4 encode error: {}", e));
                        encoder = None;
                    }
                }

                done += 1;
                let cur_pts = gpu_opt
                    .as_ref()
                    .map_or(render_solver.particles.len(), |gs| gs.particle_count);
                if let Ok(mut g) = state_arc.lock() {
                    g.anim_frames.push(preview);
                    g.completed_frames = done;
                    g.elapsed_secs = t_start.elapsed().as_secs_f32();
                    g.sim_particles_count = cur_pts;
                    g.phase_label = format!("Rendering frame {} of {}", done, total_frames);
                    g.frame_revision = g.frame_revision.wrapping_add(1);
                }
                if let Some(ref c) = ctx {
                    c.request_repaint();
                }
            }

            let saved_mp4 = if let Some(enc) = encoder {
                match enc.finish() {
                    Ok(_) if done > 0 => Some(out_path.clone()),
                    Ok(_) => None,
                    Err(e) => {
                        mp4_warn = Some(format!("FFmpeg finalize error: {}", e));
                        None
                    }
                }
            } else {
                None
            };

            let elapsed = t_start.elapsed().as_secs_f32();
            if let Ok(mut g) = state_arc.lock() {
                g.is_running = false;
                g.completed_frames = done;
                g.elapsed_secs = elapsed;
                g.phase_label = "Render complete".to_string();
                g.saved_path = saved_mp4.clone();
                if done > 0 {
                    g.full_res_frame = Some(PreviewFrameRgb {
                        width,
                        height,
                        rgb: Arc::new(off.current_rgb().to_vec()),
                    });
                    g.just_finished_anim = true;
                }
                g.finished_message = Some(if let Some(ref p) = saved_mp4 {
                    format!(
                        "Rendered {} frames in {:.1}s [{}] -> {}",
                        done, elapsed, backend_lbl, p
                    )
                } else if let Some(w) = mp4_warn {
                    format!(
                        "Rendered {} frames in {:.1}s [{}] ({})",
                        done, elapsed, backend_lbl, w
                    )
                } else {
                    format!(
                        "Rendered {} frames in {:.1}s [{}]",
                        done, elapsed, backend_lbl
                    )
                });
            }
            if let Some(c) = ctx {
                c.request_repaint();
            }
        });
    }

    fn baseline_solver_config(&self) -> SolverConfig {
        let mut base = SolverConfig::default();
        if self.current_quality != SimulationQuality::Custom {
            self.current_quality.apply(&mut base, self.is_2d);
        }
        match self.current_preset {
            ScenePreset::DamBreak => {
                base.gravity = Vec3::new(0.0, -9.81, 0.0);
                base.viscosity = 0.08;
                base.tensile_k = 0.0012;
            }
            ScenePreset::DoubleDamBreak => {
                base.gravity = Vec3::new(0.0, -9.81, 0.0);
                base.viscosity = 0.08;
                base.tensile_k = 0.0015;
            }
            ScenePreset::DropletSplash => {
                base.gravity = Vec3::new(0.0, -9.81, 0.0);
                base.viscosity = 0.05;
                base.tensile_k = 0.003;
            }
            ScenePreset::ObstacleDrop => {
                base.gravity = Vec3::new(0.0, -9.81, 0.0);
                base.viscosity = 0.08;
            }
            ScenePreset::VortexTornado => {
                base.gravity = Vec3::new(0.0, -4.0, 0.0);
                base.viscosity = 0.04;
            }
            ScenePreset::ZeroGravityBlobs => {
                base.gravity = Vec3::ZERO;
                base.viscosity = 0.15;
                base.tensile_k = 0.005;
            }
            ScenePreset::Empty => {
                base.gravity = Vec3::new(0.0, -9.81, 0.0);
                base.viscosity = 0.05;
                base.tensile_k = 0.002;
            }
        }
        if self.fluid_preset != FluidPhysicsPreset::Water
            && self.fluid_preset != FluidPhysicsPreset::Custom
            && !(self.current_preset == ScenePreset::ZeroGravityBlobs
                && self.fluid_preset == FluidPhysicsPreset::ZeroG)
        {
            self.fluid_preset.apply(&mut base);
        }
        base
    }

    /// Construct the exact CLI command string representing the active scene, physics, and surface meshing setup.
    pub fn generate_cli_command(&self) -> String {
        let mut parts = vec!["liquid-rust".to_string()];

        // Mode specific flags
        match self.cli_mode {
            CliCommandMode::Gui => {}
            CliCommandMode::Headless => {
                parts.push("--headless".to_string());
                parts.push(format!("--time {:.1}", self.cli_duration));
                parts.push(format!("--fps {}", self.cli_fps));
            }
            CliCommandMode::Mp4Video => {
                let out = if self.cli_mp4_output.is_empty() {
                    "simulation.mp4"
                } else {
                    &self.cli_mp4_output
                };
                parts.push(format!("--export-mp4 {}", out));
                parts.push(format!("--time {:.1}", self.cli_duration));
                parts.push(format!("--fps {}", self.cli_fps));
                if self.video_width != 1280 || self.video_height != 720 {
                    parts.push(format!("--video-width {}", self.video_width));
                    parts.push(format!("--video-height {}", self.video_height));
                }
                if self.gui_render_quality != CliRenderQuality::Standard {
                    parts.push(format!(
                        "--render-quality {}",
                        self.gui_render_quality.slug()
                    ));
                }
            }
            CliCommandMode::PngFrame => {
                let out = if self.png_output_path.trim().is_empty() {
                    "snapshot.png"
                } else {
                    self.png_output_path.trim()
                };
                parts.push(format!("--export-png {}", out));
                parts.push(format!("--time {:.1}", self.cli_duration));
                parts.push(format!("--fps {}", self.cli_fps));
                if self.video_width != 1280 || self.video_height != 720 {
                    parts.push(format!("--video-width {}", self.video_width));
                    parts.push(format!("--video-height {}", self.video_height));
                }
                if self.gui_render_quality != CliRenderQuality::Standard {
                    parts.push(format!(
                        "--render-quality {}",
                        self.gui_render_quality.slug()
                    ));
                }
            }
            CliCommandMode::PlySequence => {
                parts.push("--headless".to_string());
                let dir = if self.export_path.is_empty() {
                    "./ply_renders"
                } else {
                    &self.export_path
                };
                parts.push(format!("--export-ply {}", dir));
                parts.push(format!("--time {:.1}", self.cli_duration));
                parts.push(format!("--fps {}", self.cli_fps));
            }
        }

        if self.is_2d {
            parts.push("--2d".to_string());
        }

        // Preset
        let preset_slug = match self.current_preset {
            ScenePreset::DamBreak => "dam-break",
            ScenePreset::DoubleDamBreak => "double-dam",
            ScenePreset::DropletSplash => "splash",
            ScenePreset::ObstacleDrop => "obstacle",
            ScenePreset::VortexTornado => "tornado",
            ScenePreset::ZeroGravityBlobs => "zero-g",
            ScenePreset::Empty => "empty",
        };
        parts.push(format!("--preset {}", preset_slug));

        // Quality
        let quality_slug = match self.current_quality {
            SimulationQuality::Custom => None,
            other => Some(other.slug()),
        };
        if let Some(q) = quality_slug {
            parts.push(format!("--quality {}", q));
        } else {
            parts.push(format!(
                "--particle-radius {:.4}",
                self.solver.config.particle_radius
            ));
            parts.push(format!("--spacing {:.4}", self.solver.config.spacing));
        }

        // Simulation Speed / ASMR Slow-mo
        if (self.sim_speed - 1.0).abs() > 0.02 {
            parts.push(format!("--sim-speed {:.2}", self.sim_speed));
        }

        let mc = &self.mesh_config;
        let def = SurfaceMeshConfig::default();

        // Surface Meshing vs Particle shading flags
        if self.render_mode == RenderMode::SurfaceMesh {
            parts.push("--surface-mesh".to_string());

            parts.push(format!("--mesh-material {}", mc.preset.slug()));

            let mut preset_ref = SurfaceMeshConfig::default();
            mc.preset.apply(&mut preset_ref);

            if mc.preset == FluidMaterialPreset::Custom
                || (Vec3::from_array(mc.shallow_color) - Vec3::from_array(preset_ref.shallow_color))
                    .length()
                    > 0.015
            {
                parts.push(format!(
                    "--mesh-color \"{}\"",
                    SurfaceMeshConfig::format_hex(mc.shallow_color)
                ));
            }
            if mc.preset == FluidMaterialPreset::Custom
                || (Vec3::from_array(mc.deep_color) - Vec3::from_array(preset_ref.deep_color))
                    .length()
                    > 0.015
            {
                parts.push(format!(
                    "--mesh-deep-color \"{}\"",
                    SurfaceMeshConfig::format_hex(mc.deep_color)
                ));
            }
            if (mc.opacity - preset_ref.opacity).abs() > 0.02 {
                parts.push(format!("--mesh-opacity {:.2}", mc.opacity));
            }
            if (mc.absorption - preset_ref.absorption).abs() > 0.05 {
                parts.push(format!("--mesh-absorption {:.2}", mc.absorption));
            }
            if (mc.foam_intensity - preset_ref.foam_intensity).abs() > 0.02 {
                parts.push(format!("--mesh-foam {:.2}", mc.foam_intensity));
            }

            parts.push(format!("--mesh-texture {}", mc.texture_mode.slug()));
            if mc.texture_mode != MeshTextureMode::None {
                if (mc.texture_scale - def.texture_scale).abs() > 0.1 {
                    parts.push(format!("--mesh-texture-scale {:.2}", mc.texture_scale));
                }
                if (mc.texture_strength - def.texture_strength).abs() > 0.02 {
                    parts.push(format!(
                        "--mesh-texture-strength {:.2}",
                        mc.texture_strength
                    ));
                }
            }

            if (mc.ior - preset_ref.ior).abs() > 0.01 {
                parts.push(format!("--mesh-ior {:.3}", mc.ior));
            }
            if (mc.roughness - preset_ref.roughness).abs() > 0.01 {
                parts.push(format!("--mesh-roughness {:.3}", mc.roughness));
            }
            if (mc.specular_intensity - preset_ref.specular_intensity).abs() > 0.03 {
                parts.push(format!("--mesh-specular {:.2}", mc.specular_intensity));
            }
            if (mc.refraction_strength - preset_ref.refraction_strength).abs() > 0.03 {
                parts.push(format!("--mesh-refraction {:.2}", mc.refraction_strength));
            }
            if (mc.subsurface_scattering - preset_ref.subsurface_scattering).abs() > 0.03 {
                parts.push(format!("--mesh-sss {:.2}", mc.subsurface_scattering));
            }
            if (mc.env_reflection - preset_ref.env_reflection).abs() > 0.03 {
                parts.push(format!("--mesh-reflection {:.2}", mc.env_reflection));
            }
            if (mc.metallic - preset_ref.metallic).abs() > 0.02 {
                parts.push(format!("--mesh-metallic {:.2}", mc.metallic));
            }
            if (mc.voxel_scale - def.voxel_scale).abs() > 0.02 {
                parts.push(format!("--mesh-scale {:.2}", mc.voxel_scale));
            }
            if (mc.iso_value - def.iso_value).abs() > 0.02 {
                parts.push(format!("--mesh-iso {:.2}", mc.iso_value));
            }
            if mc.smoothing_passes != def.smoothing_passes {
                parts.push(format!("--mesh-smooth {}", mc.smoothing_passes));
            }
            if !mc.cast_shadows {
                parts.push("--no-shadows".to_string());
            }
        } else {
            if self.color_mode != ColorMode::Velocity {
                parts.push(format!("--color-mode {}", self.color_mode.slug()));
            }
            if self.color_map != ColorMapType::Ocean {
                parts.push(format!("--color-map {}", self.color_map.slug()));
            }
            if (self.display_size_factor - 1.25).abs() > 0.02 {
                parts.push(format!("--particle-scale {:.2}", self.display_size_factor));
            }
            if !self.particle_foam {
                parts.push("--no-particle-foam".to_string());
            }
            if !self.lit_spheres {
                parts.push("--unlit-spheres".to_string());
            }
        }

        // Studio environment & lighting overrides (apply to both Spheres and SurfaceMesh!)
        if mc.env_preset != EnvironmentPreset::StudioDark {
            parts.push(format!("--env-preset {}", mc.env_preset.slug()));
        }
        if (mc.exposure - 1.0).abs() > 0.03 {
            parts.push(format!("--exposure {:.2}", mc.exposure));
        }
        if !mc.show_grid {
            parts.push("--no-grid".to_string());
        }
        if !mc.show_wireframe {
            parts.push("--no-wireframe".to_string());
        }
        if (mc.light_dir - def.light_dir).length() > 0.03 {
            parts.push(format!(
                "--light-dir {:.2},{:.2},{:.2}",
                mc.light_dir.x, mc.light_dir.y, mc.light_dir.z
            ));
        }
        if (mc.light_intensity - def.light_intensity).abs() > 0.03 {
            parts.push(format!("--light-intensity {:.2}", mc.light_intensity));
        }
        if (mc.ambient_intensity - def.ambient_intensity).abs() > 0.03 {
            parts.push(format!("--ambient {:.2}", mc.ambient_intensity));
        }

        // Fluid substance preset & non-default physics parameters
        if self.fluid_preset != FluidPhysicsPreset::Water
            && self.fluid_preset != FluidPhysicsPreset::Custom
            && !(self.current_preset == ScenePreset::ZeroGravityBlobs
                && self.fluid_preset == FluidPhysicsPreset::ZeroG)
        {
            parts.push(format!("--fluid-preset {}", self.fluid_preset.slug()));
        }

        let base_phys = self.baseline_solver_config();
        let cfg = &self.solver.config;

        if (cfg.viscosity - base_phys.viscosity).abs() > 0.005 {
            parts.push(format!("--viscosity {:.3}", cfg.viscosity));
        }
        if (cfg.surface_tension - base_phys.surface_tension).abs() > 0.005 {
            parts.push(format!("--surface-tension {:.3}", cfg.surface_tension));
        }
        if (cfg.vorticity - base_phys.vorticity).abs() > 0.005 {
            parts.push(format!("--vorticity {:.3}", cfg.vorticity));
        }
        if (cfg.air_drag - base_phys.air_drag).abs() > 0.015 {
            parts.push(format!("--air-drag {:.2}", cfg.air_drag));
        }
        if (cfg.gravity - base_phys.gravity).length() > 0.05 {
            if cfg.gravity.x.abs() < 0.01 && cfg.gravity.z.abs() < 0.01 {
                parts.push(format!("--gravity {:.2}", cfg.gravity.y));
            } else {
                parts.push(format!(
                    "--gravity {:.2},{:.2},{:.2}",
                    cfg.gravity.x, cfg.gravity.y, cfg.gravity.z
                ));
            }
        }
        if (cfg.restitution - base_phys.restitution).abs() > 0.01 {
            parts.push(format!("--restitution {:.2}", cfg.restitution));
        }
        if (cfg.boundary_friction - base_phys.boundary_friction).abs() > 0.01 {
            parts.push(format!("--friction {:.2}", cfg.boundary_friction));
        }
        if cfg.substeps != base_phys.substeps {
            parts.push(format!("--substeps {}", cfg.substeps));
        }
        if cfg.solver_iterations != base_phys.solver_iterations {
            parts.push(format!("--iterations {}", cfg.solver_iterations));
        }
        if (cfg.tensile_k - base_phys.tensile_k).abs() > 0.0005 {
            parts.push(format!("--tensile-k {:.4}", cfg.tensile_k));
        }
        if (cfg.max_velocity - base_phys.max_velocity).abs() > 0.2 {
            parts.push(format!("--max-velocity {:.1}", cfg.max_velocity));
        }
        if cfg.wave_amplitude > 0.002 {
            parts.push(format!("--wave-amp {:.3}", cfg.wave_amplitude));
            if (cfg.wave_frequency - 0.85).abs() > 0.02 {
                parts.push(format!("--wave-freq {:.2}", cfg.wave_frequency));
            }
        }

        // Non-default camera framing
        let def_cam = Camera::default();
        if !self.is_2d {
            if self.cli_mode == CliCommandMode::Mp4Video
                || self.cli_mode == CliCommandMode::PngFrame
                || self.camera_orbit_speed_deg.abs() > 0.1
            {
                if (self.camera.azimuth - def_cam.azimuth).abs() > 0.02 {
                    parts.push(format!(
                        "--camera-azimuth {:.1}",
                        self.camera.azimuth.to_degrees()
                    ));
                }
                if (self.camera.elevation - def_cam.elevation).abs() > 0.02 {
                    parts.push(format!(
                        "--camera-elevation {:.1}",
                        self.camera.elevation.to_degrees()
                    ));
                }
                if (self.camera.distance - def_cam.distance).abs() > 0.05 {
                    parts.push(format!("--camera-distance {:.2}", self.camera.distance));
                }
            }
            if self.camera_orbit_speed_deg.abs() > 0.1 {
                parts.push(format!("--camera-orbit {:.1}", self.camera_orbit_speed_deg));
            }
        }
        if (self.camera.fov_y.to_degrees() - 45.0).abs() > 0.5 {
            parts.push(format!("--fov {:.1}", self.camera.fov_y.to_degrees()));
        }

        // Chamber shape, size & scale
        if self.solver.chamber.shape != ChamberShape::Box {
            parts.push(format!(
                "--chamber-shape {}",
                self.solver.chamber.shape.slug()
            ));
        }

        if self.solver.chamber.custom_size {
            let bs = self.solver.chamber.base_size;
            if self.is_2d {
                parts.push(format!("--chamber-size {:.2},{:.2}", bs.x, bs.y));
            } else {
                parts.push(format!(
                    "--chamber-size {:.2},{:.2},{:.2}",
                    bs.x, bs.y, bs.z
                ));
            }
        }

        let scale = self.solver.chamber.scale;
        if (scale - Vec3::ONE).length_squared() > 1e-4 {
            if (scale.x - scale.y).abs() < 1e-3 && (scale.x - scale.z).abs() < 1e-3 {
                parts.push(format!("--chamber-scale {:.2}", scale.x));
            } else {
                parts.push(format!(
                    "--chamber-scale {:.2},{:.2},{:.2}",
                    scale.x, scale.y, scale.z
                ));
            }
        }

        if self.solver.chamber.shape == ChamberShape::Bathtub {
            if (self.solver.chamber.tub_slope_ratio - 0.28).abs() > 0.005 {
                parts.push(format!(
                    "--tub-slope {:.2}",
                    self.solver.chamber.tub_slope_ratio
                ));
            }
            if !self.is_2d && (self.solver.chamber.tub_fillet_ratio - 0.16).abs() > 0.005 {
                parts.push(format!(
                    "--tub-fillet {:.2}",
                    self.solver.chamber.tub_fillet_ratio
                ));
            }
        }

        // Obstacles
        for col in &self.solver.colliders {
            if !col.enabled {
                continue;
            }
            let phys_suffix = if col.restitution.abs() > 0.005 || (col.friction - 0.02).abs() > 0.005 {
                format!(":{:.2}:{:.2}", col.restitution, col.friction)
            } else {
                String::new()
            };
            match &col.shape {
                ColliderShape::Sphere { center, radius } => {
                    parts.push(format!(
                        "--obstacle \"sphere:{:.2},{:.2},{:.2}:{:.2}{}\"",
                        center.x, center.y, center.z, radius, phys_suffix
                    ));
                }
                ColliderShape::Box { aabb } => {
                    let c = aabb.center();
                    let s = aabb.size();
                    parts.push(format!(
                        "--obstacle \"box:{:.2},{:.2},{:.2}:{:.2},{:.2},{:.2}{}\"",
                        c.x, c.y, c.z, s.x, s.y, s.z, phys_suffix
                    ));
                }
                ColliderShape::Cylinder {
                    center,
                    radius,
                    height,
                } => {
                    parts.push(format!(
                        "--obstacle \"cylinder:{:.2},{:.2},{:.2}:{:.2}:{:.2}{}\"",
                        center.x, center.y, center.z, radius, height, phys_suffix
                    ));
                }
                ColliderShape::Mesh(mesh) => {
                    let path = mesh.source_path.as_deref().unwrap_or("model.obj");
                    let p = mesh.position;
                    let r = mesh.rotation;
                    if r.length_squared() > 1e-6 {
                        parts.push(format!(
                            "--obstacle \"obj:{}:{:.2},{:.2},{:.2}:{:.2}:{:.1},{:.1},{:.1}\"",
                            path,
                            p.x,
                            p.y,
                            p.z,
                            mesh.scale,
                            r.x.to_degrees(),
                            r.y.to_degrees(),
                            r.z.to_degrees()
                        ));
                    } else {
                        parts.push(format!(
                            "--obstacle \"obj:{}:{:.2},{:.2},{:.2}:{:.2}\"",
                            path, p.x, p.y, p.z, mesh.scale
                        ));
                    }
                }
            }
        }

        // Faucets
        for f in &self.solver.faucets {
            if !f.enabled {
                continue;
            }
            let pos = f.position;
            match &f.shape {
                FaucetShape::Rectangle { width, depth } => {
                    if (depth - 0.08).abs() <= 0.005 || (depth - 0.05).abs() <= 0.005 {
                        parts.push(format!(
                            "--faucet \"waterfall:{:.2},{:.2},{:.2}:{:.2}:{:.0}\"",
                            pos.x, pos.y, pos.z, width, f.flow_rate
                        ));
                    } else {
                        parts.push(format!(
                            "--faucet \"waterfall:{:.2},{:.2},{:.2}:{:.2}:{:.0}:{:.2}\"",
                            pos.x, pos.y, pos.z, width, f.flow_rate, depth
                        ));
                    }
                }
                FaucetShape::Circle { radius } => {
                    if f.spray_spread > 0.08 {
                        parts.push(format!(
                            "--faucet \"shower:{:.2},{:.2},{:.2}:{:.3}:{:.0}:{:.2}\"",
                            pos.x, pos.y, pos.z, radius, f.flow_rate, f.spray_spread
                        ));
                    } else if (f.direction - Vec3::new(0.0, -2.5, 0.0)).length_squared() > 1e-2 {
                        let d = f.direction;
                        parts.push(format!(
                            "--faucet \"stream:{:.2},{:.2},{:.2}:{:.2},{:.2},{:.2}:{:.0}:{:.2}:{:.3}\"",
                            pos.x, pos.y, pos.z, d.x, d.y, d.z, f.flow_rate, f.spray_spread, radius
                        ));
                    } else {
                        parts.push(format!(
                            "--faucet \"circle:{:.2},{:.2},{:.2}:{:.3}:{:.0}\"",
                            pos.x, pos.y, pos.z, radius, f.flow_rate
                        ));
                    }
                }
                FaucetShape::Point => {
                    if (f.direction - Vec3::new(0.0, -2.0, 0.0)).length_squared() > 1e-2 {
                        let d = f.direction;
                        parts.push(format!(
                            "--faucet \"tap:{:.2},{:.2},{:.2}:0:{:.0}:{:.2},{:.2},{:.2}\"",
                            pos.x, pos.y, pos.z, f.flow_rate, d.x, d.y, d.z
                        ));
                    } else {
                        parts.push(format!(
                            "--faucet \"tap:{:.2},{:.2},{:.2}:0:{:.0}\"",
                            pos.x, pos.y, pos.z, f.flow_rate
                        ));
                    }
                }
            }
        }

        // Drains
        for d in &self.solver.drains {
            if !d.enabled {
                continue;
            }
            match d.face {
                CubeFace::Floor => {
                    if d.position.x.abs() > 1e-2 || d.position.z.abs() > 1e-2 {
                        parts.push(format!(
                            "--drain \"floor:{:.2},{:.2}:{:.2}\"",
                            d.position.x, d.position.z, d.radius
                        ));
                    } else {
                        parts.push(format!("--drain \"floor:{:.2}\"", d.radius));
                    }
                }
                CubeFace::Ceiling => {
                    if d.position.x.abs() > 1e-2 || d.position.z.abs() > 1e-2 {
                        parts.push(format!(
                            "--drain \"ceiling:{:.2},{:.2}:{:.2}\"",
                            d.position.x, d.position.z, d.radius
                        ));
                    } else {
                        parts.push(format!("--drain \"ceiling:{:.2}\"", d.radius));
                    }
                }
                CubeFace::Left => {
                    if d.position.y.abs() > 1e-2 || d.position.z.abs() > 1e-2 {
                        parts.push(format!(
                            "--drain \"left:{:.2},{:.2}:{:.2}\"",
                            d.position.y, d.position.z, d.radius
                        ));
                    } else {
                        parts.push(format!("--drain \"left:{:.2}\"", d.radius));
                    }
                }
                CubeFace::Right => {
                    if d.position.y.abs() > 1e-2 || d.position.z.abs() > 1e-2 {
                        parts.push(format!(
                            "--drain \"right:{:.2},{:.2}:{:.2}\"",
                            d.position.y, d.position.z, d.radius
                        ));
                    } else {
                        parts.push(format!("--drain \"right:{:.2}\"", d.radius));
                    }
                }
                CubeFace::Back => {
                    if d.position.x.abs() > 1e-2 || d.position.y.abs() > 1e-2 {
                        parts.push(format!(
                            "--drain \"back:{:.2},{:.2}:{:.2}\"",
                            d.position.x, d.position.y, d.radius
                        ));
                    } else {
                        parts.push(format!("--drain \"back:{:.2}\"", d.radius));
                    }
                }
                CubeFace::Front => {
                    if d.position.x.abs() > 1e-2 || d.position.y.abs() > 1e-2 {
                        parts.push(format!(
                            "--drain \"front:{:.2},{:.2}:{:.2}\"",
                            d.position.x, d.position.y, d.radius
                        ));
                    } else {
                        parts.push(format!("--drain \"front:{:.2}\"", d.radius));
                    }
                }
                CubeFace::Custom => {
                    let p = d.position;
                    parts.push(format!(
                        "--drain \"custom:{:.2},{:.2},{:.2}:{:.2}\"",
                        p.x, p.y, p.z, d.radius
                    ));
                }
            }
        }

        parts.join(" ")
    }

    fn compute_particle_color(&self, p: &crate::sph::particle::Particle) -> [u8; 3] {
        ColorMap::particle_color(
            p,
            self.color_mode,
            self.color_map,
            self.solver.config.rest_density,
            &self.solver.bounds,
            self.particle_foam,
        )
    }

    fn paint_icon(painter: &egui::Painter, center: egui::Pos2, icon: UiIcon, color: egui::Color32) {
        let s = egui::Stroke::new(1.35_f32, color);
        match icon {
            UiIcon::Wave => {
                let pts = [
                    egui::pos2(center.x - 5.5, center.y + 1.5),
                    egui::pos2(center.x - 2.8, center.y - 2.2),
                    egui::pos2(center.x, center.y + 1.5),
                    egui::pos2(center.x + 2.8, center.y - 2.2),
                    egui::pos2(center.x + 5.5, center.y + 1.5),
                ];
                for w in pts.windows(2) {
                    painter.line_segment([w[0], w[1]], s);
                }
            }
            UiIcon::Cube3D => {
                let top = [
                    egui::pos2(center.x, center.y - 5.2),
                    egui::pos2(center.x + 4.8, center.y - 2.5),
                    egui::pos2(center.x, center.y + 0.2),
                    egui::pos2(center.x - 4.8, center.y - 2.5),
                ];
                for i in 0..4 {
                    painter.line_segment([top[i], top[(i + 1) % 4]], s);
                }
                let bot_l = egui::pos2(center.x - 4.8, center.y + 2.8);
                let bot_c = egui::pos2(center.x, center.y + 5.4);
                let bot_r = egui::pos2(center.x + 4.8, center.y + 2.8);
                painter.line_segment([top[3], bot_l], s);
                painter.line_segment([top[2], bot_c], s);
                painter.line_segment([top[1], bot_r], s);
                painter.line_segment([bot_l, bot_c], s);
                painter.line_segment([bot_c, bot_r], s);
            }
            UiIcon::Plane2D => {
                let r = egui::Rect::from_center_size(center, egui::vec2(9.6, 7.6));
                painter.rect_stroke(r, egui::CornerRadius::same(1), s, egui::StrokeKind::Inside);
                painter.line_segment(
                    [egui::pos2(r.left() + 2.0, center.y + 1.0), egui::pos2(r.right() - 2.0, center.y - 1.0)],
                    egui::Stroke::new(1.1_f32, color),
                );
            }
            UiIcon::Play => {
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(center.x - 3.2, center.y - 4.5),
                        egui::pos2(center.x + 4.5, center.y),
                        egui::pos2(center.x - 3.2, center.y + 4.5),
                    ],
                    color,
                    egui::Stroke::NONE,
                ));
            }
            UiIcon::Pause => {
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(center.x - 3.8, center.y - 4.4),
                        egui::pos2(center.x - 1.2, center.y + 4.4),
                    ),
                    1.0,
                    color,
                );
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(center.x + 1.2, center.y - 4.4),
                        egui::pos2(center.x + 3.8, center.y + 4.4),
                    ),
                    1.0,
                    color,
                );
            }
            UiIcon::Step => {
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(center.x - 4.4, center.y - 4.0),
                        egui::pos2(center.x + 1.4, center.y),
                        egui::pos2(center.x - 4.4, center.y + 4.0),
                    ],
                    color,
                    egui::Stroke::NONE,
                ));
                painter.line_segment(
                    [
                        egui::pos2(center.x + 3.4, center.y - 4.2),
                        egui::pos2(center.x + 3.4, center.y + 4.2),
                    ],
                    egui::Stroke::new(1.8_f32, color),
                );
            }
            UiIcon::StepBack => {
                painter.line_segment(
                    [
                        egui::pos2(center.x - 3.4, center.y - 4.2),
                        egui::pos2(center.x - 3.4, center.y + 4.2),
                    ],
                    egui::Stroke::new(1.8_f32, color),
                );
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(center.x + 4.4, center.y - 4.0),
                        egui::pos2(center.x - 1.4, center.y),
                        egui::pos2(center.x + 4.4, center.y + 4.0),
                    ],
                    color,
                    egui::Stroke::NONE,
                ));
            }
            UiIcon::Reset => {
                painter.circle_stroke(center, 4.3, s);
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(center.x + 1.6, center.y - 5.6),
                        egui::pos2(center.x + 5.2, center.y - 4.2),
                        egui::pos2(center.x + 2.6, center.y - 1.8),
                    ],
                    color,
                    egui::Stroke::NONE,
                ));
            }
            UiIcon::Camera => {
                let body = egui::Rect::from_min_max(
                    egui::pos2(center.x - 5.2, center.y - 3.4),
                    egui::pos2(center.x + 2.2, center.y + 3.4),
                );
                painter.rect_stroke(body, egui::CornerRadius::same(1), s, egui::StrokeKind::Inside);
                painter.line_segment([egui::pos2(center.x + 2.2, center.y - 1.2), egui::pos2(center.x + 5.2, center.y - 3.0)], s);
                painter.line_segment([egui::pos2(center.x + 5.2, center.y - 3.0), egui::pos2(center.x + 5.2, center.y + 3.0)], s);
                painter.line_segment([egui::pos2(center.x + 5.2, center.y + 3.0), egui::pos2(center.x + 2.2, center.y + 1.2)], s);
            }
            UiIcon::Spheres => {
                painter.circle_stroke(egui::pos2(center.x - 1.8, center.y + 1.2), 3.3, s);
                painter.circle_filled(egui::pos2(center.x + 2.6, center.y - 2.2), 2.1, color);
            }
            UiIcon::Mesh => {
                let pts = [
                    egui::pos2(center.x, center.y - 5.0),
                    egui::pos2(center.x + 4.8, center.y),
                    egui::pos2(center.x, center.y + 5.0),
                    egui::pos2(center.x - 4.8, center.y),
                ];
                for i in 0..4 {
                    painter.line_segment([pts[i], pts[(i + 1) % 4]], s);
                }
                painter.line_segment([pts[3], pts[1]], egui::Stroke::new(1.0_f32, color));
            }
            UiIcon::Sliders => {
                for &(y_off, knob_x) in &[(-3.4f32, -1.6f32), (0.0, 2.0), (3.4, -0.8)] {
                    painter.line_segment(
                        [
                            egui::pos2(center.x - 4.8, center.y + y_off),
                            egui::pos2(center.x + 4.8, center.y + y_off),
                        ],
                        egui::Stroke::new(1.2_f32, color),
                    );
                    painter.circle_filled(egui::pos2(center.x + knob_x, center.y + y_off), 1.7, color);
                }
            }
            UiIcon::Film => {
                let r = egui::Rect::from_center_size(center, egui::vec2(10.0, 8.2));
                painter.rect_stroke(r, egui::CornerRadius::same(1), s, egui::StrokeKind::Inside);
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(center.x - 1.5, center.y - 2.3),
                        egui::pos2(center.x + 2.4, center.y),
                        egui::pos2(center.x - 1.5, center.y + 2.3),
                    ],
                    color,
                    egui::Stroke::NONE,
                ));
            }
            UiIcon::Copy => {
                let back = egui::Rect::from_min_max(
                    egui::pos2(center.x - 4.4, center.y - 4.6),
                    egui::pos2(center.x + 2.2, center.y + 2.2),
                );
                let front = egui::Rect::from_min_max(
                    egui::pos2(center.x - 2.0, center.y - 2.2),
                    egui::pos2(center.x + 4.6, center.y + 4.6),
                );
                painter.rect_stroke(back, egui::CornerRadius::same(1), s, egui::StrokeKind::Inside);
                painter.rect_stroke(front, egui::CornerRadius::same(1), s, egui::StrokeKind::Inside);
            }
            UiIcon::Folder => {
                let body = egui::Rect::from_min_max(
                    egui::pos2(center.x - 5.0, center.y - 2.4),
                    egui::pos2(center.x + 5.0, center.y + 4.2),
                );
                painter.rect_stroke(body, egui::CornerRadius::same(1), s, egui::StrokeKind::Inside);
                painter.line_segment([egui::pos2(center.x - 5.0, center.y - 2.4), egui::pos2(center.x - 4.2, center.y - 4.2)], s);
                painter.line_segment([egui::pos2(center.x - 4.2, center.y - 4.2), egui::pos2(center.x - 0.8, center.y - 4.2)], s);
                painter.line_segment([egui::pos2(center.x - 0.8, center.y - 4.2), egui::pos2(center.x + 0.4, center.y - 2.4)], s);
            }
            UiIcon::Save => {
                let r = egui::Rect::from_center_size(center, egui::vec2(9.2, 9.2));
                painter.rect_stroke(r, egui::CornerRadius::same(1), s, egui::StrokeKind::Inside);
                let top = egui::Rect::from_min_max(
                    egui::pos2(center.x - 2.6, r.top() + 0.5),
                    egui::pos2(center.x + 2.4, center.y - 1.2),
                );
                painter.rect_stroke(top, egui::CornerRadius::ZERO, egui::Stroke::new(1.0_f32, color), egui::StrokeKind::Inside);
            }
            UiIcon::Image => {
                let r = egui::Rect::from_center_size(center, egui::vec2(10.0, 8.0));
                painter.rect_stroke(r, egui::CornerRadius::same(1), s, egui::StrokeKind::Inside);
                painter.circle_filled(egui::pos2(r.left() + 2.8, r.top() + 2.5), 1.1, color);
                painter.line_segment([egui::pos2(r.left() + 1.5, r.bottom() - 1.2), egui::pos2(center.x, center.y - 0.4)], s);
                painter.line_segment([egui::pos2(center.x, center.y - 0.4), egui::pos2(r.right() - 1.4, r.bottom() - 1.2)], s);
            }
            UiIcon::Help => {
                painter.circle_stroke(center, 5.0, s);
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    "?",
                    egui::FontId::proportional(10.5),
                    color,
                );
            }
            UiIcon::Warning => {
                painter.add(egui::Shape::closed_line(
                    vec![
                        egui::pos2(center.x, center.y - 5.0),
                        egui::pos2(center.x + 5.2, center.y + 4.4),
                        egui::pos2(center.x - 5.2, center.y + 4.4),
                    ],
                    s,
                ));
                painter.line_segment(
                    [egui::pos2(center.x, center.y - 1.8), egui::pos2(center.x, center.y + 1.2)],
                    s,
                );
                painter.circle_filled(egui::pos2(center.x, center.y + 2.8), 0.9, color);
            }
            UiIcon::Bolt => {
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(center.x + 0.8, center.y - 5.2),
                        egui::pos2(center.x - 3.8, center.y + 0.4),
                        egui::pos2(center.x - 0.2, center.y + 0.4),
                        egui::pos2(center.x - 1.0, center.y + 5.2),
                        egui::pos2(center.x + 3.8, center.y - 0.6),
                        egui::pos2(center.x + 0.2, center.y - 0.6),
                    ],
                    color,
                    egui::Stroke::NONE,
                ));
            }
        }
    }

    fn icon_button(
        ui: &mut egui::Ui,
        icon: UiIcon,
        label: &str,
        selected: bool,
        min_size: Option<egui::Vec2>,
    ) -> egui::Response {
        let font_id = egui::TextStyle::Button.resolve(ui.style());
        let galley = if label.is_empty() {
            None
        } else {
            Some(ui.fonts(|f| {
                f.layout_no_wrap(label.to_string(), font_id.clone(), egui::Color32::WHITE)
            }))
        };
        let text_w = galley.as_ref().map_or(0.0, |g| g.size().x);
        let gap = if label.is_empty() { 0.0 } else { 5.5 };
        let icon_w = 13.0;
        let pad = ui.spacing().button_padding;
        let auto_size = egui::vec2(icon_w + gap + text_w + pad.x * 2.0, 24.0);
        let desired = min_size.map_or(auto_size, |m| auto_size.max(m));

        let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact_selectable(&response, selected);
            let bg = if selected {
                egui::Color32::from_rgb(20, 110, 200)
            } else {
                visuals.bg_fill
            };
            let stroke = if selected {
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(88, 184, 255))
            } else {
                visuals.bg_stroke
            };
            let fg = if selected {
                egui::Color32::WHITE
            } else {
                visuals.fg_stroke.color
            };

            let p = ui.painter();
            p.rect(
                rect,
                visuals.corner_radius,
                bg,
                stroke,
                egui::StrokeKind::Inside,
            );

            let content_w = icon_w + gap + text_w;
            let start_x = rect.center().x - content_w * 0.5;
            let icon_center = egui::pos2(start_x + icon_w * 0.5, rect.center().y);
            Self::paint_icon(p, icon_center, icon, fg);

            if !label.is_empty() {
                p.text(
                    egui::pos2(start_x + icon_w + gap, rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    label,
                    font_id,
                    fg,
                );
            }
        }
        response
    }

    fn inspector_tab_button(
        ui: &mut egui::Ui,
        icon: UiIcon,
        label: &str,
        selected: bool,
        width: f32,
    ) -> egui::Response {
        let font_id = egui::FontId::proportional(11.5);
        let galley = ui.fonts(|f| {
            f.layout_no_wrap(label.to_string(), font_id.clone(), egui::Color32::WHITE)
        });
        let text_w = galley.size().x;
        let icon_w = 11.5;
        let gap = 4.0;
        let desired = egui::vec2(width, 26.0);

        let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact_selectable(&response, selected);
            let bg = if selected {
                egui::Color32::from_rgb(20, 110, 200)
            } else {
                visuals.bg_fill
            };
            let stroke = if selected {
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(88, 184, 255))
            } else {
                visuals.bg_stroke
            };
            let fg = if selected {
                egui::Color32::WHITE
            } else {
                visuals.fg_stroke.color
            };

            let p = ui.painter();
            p.rect(
                rect,
                visuals.corner_radius,
                bg,
                stroke,
                egui::StrokeKind::Inside,
            );

            let content_w = icon_w + gap + text_w;
            let start_x = rect.center().x - content_w * 0.5;
            let icon_center = egui::pos2(start_x + icon_w * 0.5, rect.center().y);
            Self::paint_icon(p, icon_center, icon, fg);
            p.text(
                egui::pos2(start_x + icon_w + gap, rect.center().y),
                egui::Align2::LEFT_CENTER,
                label,
                font_id,
                fg,
            );
        }
        response
    }

    fn draw_telemetry_pill(ui: &mut egui::Ui, text: String, color: egui::Color32) {
        egui::Frame::NONE
            .fill(egui::Color32::from_rgb(15, 21, 33))
            .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(33, 45, 67)))
            .corner_radius(egui::CornerRadius::same(5))
            .inner_margin(egui::Margin::symmetric(8, 3))
            .show(ui, |ui| {
                ui.label(egui::RichText::new(text).monospace().size(11.5).color(color));
            });
    }

    fn section_card<R>(
        ui: &mut egui::Ui,
        title: &str,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> R {
        egui::Frame::NONE
            .fill(egui::Color32::from_rgb(14, 20, 32))
            .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(29, 40, 60)))
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(
                    egui::RichText::new(title)
                        .strong()
                        .size(12.2)
                        .color(egui::Color32::from_rgb(138, 192, 255)),
                );
                ui.add_space(4.0);
                add_contents(ui)
            })
            .inner
    }

    fn trash_icon_button(ui: &mut egui::Ui, tooltip: &str) -> bool {
        let (rect, response) = ui.allocate_exact_size(egui::vec2(22.0, 20.0), egui::Sense::click());
        let hovered = response.hovered();
        let active = response.is_pointer_button_down_on();

        if ui.is_rect_visible(rect) {
            let (bg, border, icon_col) = if active {
                (
                    egui::Color32::from_rgb(92, 32, 42),
                    egui::Color32::from_rgb(225, 95, 105),
                    egui::Color32::from_rgb(255, 165, 170),
                )
            } else if hovered {
                (
                    egui::Color32::from_rgb(68, 28, 38),
                    egui::Color32::from_rgb(196, 82, 92),
                    egui::Color32::from_rgb(252, 132, 138),
                )
            } else {
                (
                    egui::Color32::from_rgb(24, 34, 52),
                    egui::Color32::from_rgb(46, 63, 92),
                    egui::Color32::from_rgb(176, 192, 214),
                )
            };

            let p = ui.painter();
            p.rect(
                rect,
                egui::CornerRadius::same(4),
                bg,
                egui::Stroke::new(1.0_f32, border),
                egui::StrokeKind::Inside,
            );

            let c = rect.center();
            let s = egui::Stroke::new(1.25_f32, icon_col);
            // Lid bar
            p.line_segment([egui::pos2(c.x - 4.8, c.y - 3.0), egui::pos2(c.x + 4.8, c.y - 3.0)], s);
            // Top handle
            p.line_segment([egui::pos2(c.x - 1.9, c.y - 3.0), egui::pos2(c.x - 1.9, c.y - 4.9)], s);
            p.line_segment([egui::pos2(c.x - 1.9, c.y - 4.9), egui::pos2(c.x + 1.9, c.y - 4.9)], s);
            p.line_segment([egui::pos2(c.x + 1.9, c.y - 4.9), egui::pos2(c.x + 1.9, c.y - 3.0)], s);
            // Tapered bin walls & base
            p.line_segment([egui::pos2(c.x - 3.7, c.y - 3.0), egui::pos2(c.x - 2.9, c.y + 4.8)], s);
            p.line_segment([egui::pos2(c.x - 2.9, c.y + 4.8), egui::pos2(c.x + 2.9, c.y + 4.8)], s);
            p.line_segment([egui::pos2(c.x + 2.9, c.y + 4.8), egui::pos2(c.x + 3.7, c.y - 3.0)], s);
            // Inner vertical ribs
            p.line_segment([egui::pos2(c.x - 1.25, c.y - 1.2), egui::pos2(c.x - 1.05, c.y + 3.2)], s);
            p.line_segment([egui::pos2(c.x + 1.25, c.y - 1.2), egui::pos2(c.x + 1.05, c.y + 3.2)], s);
        }

        response.on_hover_text(tooltip).clicked()
    }

    fn draw_scene_tab(&mut self, ui: &mut egui::Ui) {
        // Consume async load/save scene file dialog selections
        if let Some(load_path) = self
            .scene_load_result
            .lock()
            .ok()
            .and_then(|mut g| g.take())
        {
            self.load_scene_json(&load_path);
        }
        if let Some(save_path) = self
            .scene_save_result
            .lock()
            .ok()
            .and_then(|mut g| g.take())
        {
            self.save_scene_json(&save_path);
        }

        // 0. Simulation Space & Studio Scene File (.json)
        Self::section_card(ui, "Simulation Space & Studio Project", |ui| {
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 6.0) * 0.5;
                if Self::icon_button(ui, UiIcon::Cube3D, "3D Volumetric", !self.is_2d, Some(egui::vec2(w, 26.0)))
                    .clicked()
                    && self.is_2d
                {
                    self.set_2d(false);
                }
                if Self::icon_button(ui, UiIcon::Plane2D, "2D Planar", self.is_2d, Some(egui::vec2(w, 26.0)))
                    .clicked()
                    && !self.is_2d
                {
                    self.set_2d(true);
                }
            });

            ui.add_space(3.0);
            let scene_busy = self.scene_browse_pending.load(Ordering::Relaxed);
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 6.0) * 0.5;
                ui.add_enabled_ui(!scene_busy, |ui| {
                    if Self::icon_button(
                        ui,
                        UiIcon::Folder,
                        "Load Scene (.json)",
                        false,
                        Some(egui::vec2(w, 24.0)),
                    )
                    .on_hover_text("Open a saved LiquidRust studio scene (.json)")
                    .clicked()
                    {
                        self.spawn_scene_file_dialog(ui.ctx().clone(), false);
                    }
                    if Self::icon_button(
                        ui,
                        UiIcon::Save,
                        "Save Scene (.json)",
                        false,
                        Some(egui::vec2(w, 24.0)),
                    )
                    .on_hover_text("Save current scene, physics, obstacles, faucets, drains & visuals to .json")
                    .clicked()
                    {
                        self.spawn_scene_file_dialog(ui.ctx().clone(), true);
                    }
                });
            });
        });

        ui.add_space(6.0);

        // 1. Scene Presets
        Self::section_card(ui, "Scene Presets", |ui| {
            let col_w = ((ui.available_width() - 6.0) * 0.5).max(132.0);
            egui::Grid::new("preset_grid")
                .num_columns(2)
                .spacing([6.0, 5.0])
                .show(ui, |ui| {
                    for (i, &preset) in ScenePreset::ALL.iter().enumerate() {
                        let is_cur = self.current_preset == preset;
                        if ui
                            .add_sized(
                                [col_w, 24.0],
                                egui::Button::new(preset.name()).selected(is_cur),
                            )
                            .clicked()
                        {
                            self.load_preset(preset);
                        }
                        if i % 2 == 1 {
                            ui.end_row();
                        }
                    }
                });
        });

        ui.add_space(6.0);

        // 2. Particle Quality & Resolution
        Self::section_card(ui, "Particle Resolution", |ui| {
            for &q in SimulationQuality::ALL {
                let is_cur = self.current_quality == q;
                let label_text = if q == SimulationQuality::Extreme {
                    egui::RichText::new(q.name_for(self.is_2d))
                        .color(egui::Color32::from_rgb(255, 150, 72))
                } else {
                    egui::RichText::new(q.name_for(self.is_2d))
                };
                if ui.selectable_label(is_cur, label_text).clicked() {
                    if !self.is_2d && q == SimulationQuality::Extreme && !is_cur {
                        self.show_extreme_modal = true;
                    } else {
                        self.set_quality(q);
                    }
                }
            }

            egui::CollapsingHeader::new("Advanced Lattice Overrides")
                .default_open(self.current_quality == SimulationQuality::Custom)
                .show(ui, |ui| {
                    let mut changed = false;
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.solver.config.particle_radius, 0.004..=0.045)
                                .text("Radius")
                                .suffix(" m"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.solver.config.spacing, 0.007..=0.070)
                                .text("Spacing")
                                .suffix(" m"),
                        )
                        .changed();

                    if changed {
                        self.current_quality = SimulationQuality::Custom;
                        self.solver.config.smoothing_length =
                            self.solver.config.particle_radius * 3.1;
                        self.solver.config.rest_density = if self.is_2d {
                            compute_lattice_rest_density_2d(
                                self.solver.config.smoothing_length,
                                self.solver.config.spacing,
                            )
                        } else {
                            compute_lattice_rest_density(
                                self.solver.config.smoothing_length,
                                self.solver.config.spacing,
                            )
                        };
                    }

                    if Self::icon_button(ui, UiIcon::Reset, "Respawn Lattice", false, None).clicked()
                    {
                        let h = self.solver.config.smoothing_length;
                        self.solver.set_smoothing_length(h);
                        self.load_preset(self.current_preset);
                    }
                });
        });

        ui.add_space(6.0);

        // 3. Simulation Chamber, Exact Dimensions & Wave Maker
        Self::section_card(ui, "Container Chamber & Wave Maker", |ui| {
            ui.horizontal(|ui| {
                for &s in &ChamberShape::ALL {
                    if ui
                        .selectable_label(self.solver.chamber.shape == s, s.name())
                        .clicked()
                    {
                        self.solver.chamber.shape = s;
                        self.solver.bounds = self.solver.chamber.compute_bounds();
                        if let Some(ref mut gpu) = self.gpu_solver {
                            gpu.sync_particles(
                                self.solver.bounds,
                                &self.solver.config,
                                &self.solver.particles,
                                false,
                            );
                        }
                    }
                }
            });

            let mut bounds_changed = false;
            let mut uniform_s = (self.solver.chamber.scale.x
                + self.solver.chamber.scale.y
                + self.solver.chamber.scale.z)
                / 3.0;
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Slider::new(&mut uniform_s, 0.4..=3.0).text("Scale"))
                    .changed()
                {
                    self.solver.chamber.scale = Vec3::splat(uniform_s);
                    bounds_changed = true;
                }
                if ui
                    .small_button("1.0x")
                    .on_hover_text("Reset chamber scale & custom size")
                    .clicked()
                {
                    self.solver.chamber.scale = Vec3::ONE;
                    self.solver.chamber.base_size = Vec3::new(1.6, 1.2, 0.9);
                    self.solver.chamber.custom_size = false;
                    bounds_changed = true;
                }
            });

            // Wave-maker paddle right in chamber card (compact 1-line activation, expands frequency when > 0)
            ui.add(
                egui::Slider::new(&mut self.solver.config.wave_amplitude, 0.0..=0.45)
                    .text("Wave Maker")
                    .suffix(" m")
                    .fixed_decimals(2),
            )
            .on_hover_text("Oscillating hydraulic wave paddle along the left chamber wall (0.0 = off)");
            if self.solver.config.wave_amplitude > 0.002 {
                ui.add(
                    egui::Slider::new(&mut self.solver.config.wave_frequency, 0.2..=3.5)
                        .text("Wave Freq")
                        .suffix(" Hz")
                        .fixed_decimals(2),
                );
            }

            egui::CollapsingHeader::new("Advanced Chamber Dimensions & Contours")
                .default_open(
                    self.solver.chamber.custom_size
                        || self.solver.chamber.shape == ChamberShape::Bathtub,
                )
                .show(ui, |ui| {
                    bounds_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.solver.chamber.scale.x, 0.3..=3.5)
                                .text("Scale X"),
                        )
                        .changed();
                    bounds_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.solver.chamber.scale.y, 0.3..=3.5)
                                .text("Scale Y"),
                        )
                        .changed();
                    if !self.is_2d {
                        bounds_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.solver.chamber.scale.z, 0.3..=3.5)
                                    .text("Scale Z"),
                            )
                            .changed();
                    }

                    ui.separator();
                    ui.label("Base Size (meters):");
                    let mut size_edited = false;
                    size_edited |= ui
                        .add(
                            egui::Slider::new(&mut self.solver.chamber.base_size.x, 0.4..=5.0)
                                .text("Width X")
                                .suffix(" m"),
                        )
                        .changed();
                    size_edited |= ui
                        .add(
                            egui::Slider::new(&mut self.solver.chamber.base_size.y, 0.4..=4.0)
                                .text("Height Y")
                                .suffix(" m"),
                        )
                        .changed();
                    if !self.is_2d {
                        size_edited |= ui
                            .add(
                                egui::Slider::new(&mut self.solver.chamber.base_size.z, 0.3..=4.0)
                                    .text("Depth Z")
                                    .suffix(" m"),
                            )
                            .changed();
                    }
                    if size_edited {
                        self.solver.chamber.custom_size = true;
                        bounds_changed = true;
                    }

                    if self.solver.chamber.shape == ChamberShape::Bathtub {
                        ui.separator();
                        ui.add(
                            egui::Slider::new(&mut self.solver.chamber.tub_slope_ratio, 0.1..=0.45)
                                .text("Backrest Slope"),
                        );
                        if !self.is_2d {
                            ui.add(
                                egui::Slider::new(
                                    &mut self.solver.chamber.tub_fillet_ratio,
                                    0.05..=0.35,
                                )
                                .text("Bottom Fillet"),
                            );
                        }
                    }
                });

            if bounds_changed {
                self.solver.bounds = self.solver.chamber.compute_bounds();
                if self.is_2d {
                    self.solver.bounds.min.z = -0.05;
                    self.solver.bounds.max.z = 0.05;
                }
            }

            let sz = self.solver.bounds.size();
            ui.weak(if self.is_2d {
                format!("Active Bounds: {:.2} × {:.2} m", sz.x, sz.y)
            } else {
                format!("Active Bounds: {:.2} × {:.2} × {:.2} m", sz.x, sz.y, sz.z)
            });
        });

        ui.add_space(6.0);

        // 4. Obstacles, Faucets & Drains
        let obs_title = format!("Obstacles & Colliders ({})", self.solver.colliders.len());
        egui::CollapsingHeader::new(obs_title)
            .default_open(!self.solver.colliders.is_empty())
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("+ Sphere").clicked() {
                        let id = self.next_collider_id;
                        self.next_collider_id += 1;
                        let mut c = Collider::sphere(Vec3::new(0.0, 0.45, 0.0), 0.18);
                        c.id = id;
                        c.name = format!("Sphere #{}", id);
                        self.solver.colliders.push(c);
                    }
                    if ui.button("+ Box").clicked() {
                        let id = self.next_collider_id;
                        self.next_collider_id += 1;
                        let mut c = if self.is_2d {
                            Collider::box_obstacle(
                                Vec3::new(-0.2, 0.05, -0.05),
                                Vec3::new(0.2, 0.35, 0.05),
                            )
                        } else {
                            Collider::box_obstacle(
                                Vec3::new(-0.2, 0.05, -0.2),
                                Vec3::new(0.2, 0.35, 0.2),
                            )
                        };
                        c.id = id;
                        c.name = format!("Box #{}", id);
                        self.solver.colliders.push(c);
                    }
                    if ui.button("+ Cylinder").clicked() {
                        let id = self.next_collider_id;
                        self.next_collider_id += 1;
                        let mut c = Collider::cylinder(Vec3::new(0.0, 0.35, 0.0), 0.15, 0.4);
                        c.id = id;
                        c.name = format!("Cylinder #{}", id);
                        self.solver.colliders.push(c);
                    }
                });

                // Consume any file chosen from the async native OBJ browser dialog
                let picked_obj = self
                    .obj_browse_result
                    .lock()
                    .ok()
                    .and_then(|mut g| g.take());
                if let Some(picked_path) = picked_obj {
                    self.obj_file_path = picked_path.clone();
                    self.load_obj_obstacle(&picked_path);
                }

                ui.horizontal(|ui| {
                    ui.label("OBJ:");
                    let text_resp = ui.add(
                        egui::TextEdit::singleline(&mut self.obj_file_path).desired_width(120.0),
                    );
                    let enter_pressed =
                        text_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                    let browsing = self.obj_browse_pending.load(Ordering::Relaxed);
                    ui.add_enabled_ui(!browsing, |ui| {
                        if Self::icon_button(ui, UiIcon::Folder, "Browse", false, None)
                            .on_hover_text("Browse your computer for a Wavefront .OBJ 3D mesh")
                            .clicked()
                        {
                            self.spawn_obj_file_browser(ui.ctx().clone());
                        }
                    });

                    if ui
                        .button("Load")
                        .on_hover_text("Load .OBJ mesh obstacle into the chamber")
                        .clicked()
                        || enter_pressed
                    {
                        let path = self.obj_file_path.trim().to_string();
                        self.load_obj_obstacle(&path);
                    }
                });

                let sim_bounds = self.solver.bounds;
                let b_center = sim_bounds.center();
                let b_size = sim_bounds.size();
                let default_span = if self.is_2d {
                    b_size.x.min(b_size.y)
                } else {
                    b_size.x.min(b_size.y).min(b_size.z)
                };
                let default_mesh_scale = (default_span * 0.28).clamp(0.15, 0.65);
                let default_mesh_pos = Vec3::new(
                    b_center.x,
                    sim_bounds.min.y + b_size.y * 0.35,
                    if self.is_2d { 0.0 } else { b_center.z },
                );

                let mut to_remove_col: Option<usize> = None;
                for (idx, collider) in self.solver.colliders.iter_mut().enumerate() {
                    ui.separator();
                    let tri_badge = if let ColliderShape::Mesh(m) = &collider.shape {
                        Some(m.triangle_count())
                    } else {
                        None
                    };
                    let mut reset_mesh_transform = false;

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut collider.enabled, "");
                        ui.strong(&collider.name);
                        if let Some(tris) = tri_badge {
                            ui.small(
                                egui::RichText::new(format!("({} tris)", tris))
                                    .color(egui::Color32::from_rgb(138, 152, 172)),
                            );
                        }
                        if tri_badge.is_some()
                            && ui
                                .small_button("Reset")
                                .on_hover_text("Reset position, scale & rotation")
                                .clicked()
                        {
                            reset_mesh_transform = true;
                        }
                        if Self::trash_icon_button(ui, "Remove obstacle") {
                            to_remove_col = Some(idx);
                        }
                    });

                    if collider.enabled {
                        match &mut collider.shape {
                            ColliderShape::Sphere { center, radius } => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut center.x,
                                        sim_bounds.min.x..=sim_bounds.max.x,
                                    )
                                    .text("Move X")
                                    .suffix(" m")
                                    .clamping(egui::SliderClamping::Edits),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut center.y,
                                        sim_bounds.min.y..=sim_bounds.max.y,
                                    )
                                    .text("Move Y")
                                    .suffix(" m")
                                    .clamping(egui::SliderClamping::Edits),
                                );
                                if !self.is_2d {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut center.z,
                                            sim_bounds.min.z..=sim_bounds.max.z,
                                        )
                                        .text("Move Z")
                                        .suffix(" m")
                                        .clamping(egui::SliderClamping::Edits),
                                    );
                                } else {
                                    center.z = 0.0;
                                }
                                ui.add(
                                    egui::Slider::new(radius, 0.02..=0.8)
                                        .text("Radius")
                                        .suffix(" m")
                                        .clamping(egui::SliderClamping::Edits),
                                );
                            }
                            ColliderShape::Box { aabb } => {
                                let mut center = aabb.center();
                                let mut extents = aabb.size();
                                let mut changed = false;
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut center.x,
                                            sim_bounds.min.x..=sim_bounds.max.x,
                                        )
                                        .text("Move X")
                                        .suffix(" m")
                                        .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut center.y,
                                            sim_bounds.min.y..=sim_bounds.max.y,
                                        )
                                        .text("Move Y")
                                        .suffix(" m")
                                        .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();
                                if !self.is_2d {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(
                                                &mut center.z,
                                                sim_bounds.min.z..=sim_bounds.max.z,
                                            )
                                            .text("Move Z")
                                            .suffix(" m")
                                            .clamping(egui::SliderClamping::Edits),
                                        )
                                        .changed();
                                } else {
                                    center.z = 0.0;
                                }
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut extents.x, 0.04..=1.6)
                                            .text("Width")
                                            .suffix(" m")
                                            .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut extents.y, 0.04..=1.6)
                                            .text("Height")
                                            .suffix(" m")
                                            .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();
                                if !self.is_2d {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(&mut extents.z, 0.04..=1.6)
                                                .text("Depth")
                                                .suffix(" m")
                                                .clamping(egui::SliderClamping::Edits),
                                        )
                                        .changed();
                                }
                                if changed {
                                    let h = extents * 0.5;
                                    *aabb = Aabb::new(center - h, center + h);
                                }
                            }
                            ColliderShape::Cylinder {
                                center,
                                radius,
                                height,
                            } => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut center.x,
                                        sim_bounds.min.x..=sim_bounds.max.x,
                                    )
                                    .text("Move X")
                                    .suffix(" m")
                                    .clamping(egui::SliderClamping::Edits),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut center.y,
                                        sim_bounds.min.y..=sim_bounds.max.y,
                                    )
                                    .text("Move Y")
                                    .suffix(" m")
                                    .clamping(egui::SliderClamping::Edits),
                                );
                                if !self.is_2d {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut center.z,
                                            sim_bounds.min.z..=sim_bounds.max.z,
                                        )
                                        .text("Move Z")
                                        .suffix(" m")
                                        .clamping(egui::SliderClamping::Edits),
                                    );
                                } else {
                                    center.z = 0.0;
                                }
                                ui.add(
                                    egui::Slider::new(radius, 0.02..=0.8)
                                        .text("Radius")
                                        .suffix(" m")
                                        .clamping(egui::SliderClamping::Edits),
                                );
                                if !self.is_2d {
                                    ui.add(
                                        egui::Slider::new(height, 0.05..=1.6)
                                            .text("Height")
                                            .suffix(" m")
                                            .clamping(egui::SliderClamping::Edits),
                                    );
                                }
                            }
                            ColliderShape::Mesh(mesh_arc) => {
                                let mesh = std::sync::Arc::make_mut(mesh_arc);
                                let mut changed = false;

                                if reset_mesh_transform {
                                    mesh.position = default_mesh_pos;
                                    mesh.scale = default_mesh_scale;
                                    mesh.rotation = Vec3::ZERO;
                                    changed = true;
                                }

                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut mesh.position.x,
                                            sim_bounds.min.x..=sim_bounds.max.x,
                                        )
                                        .text("Move X")
                                        .suffix(" m")
                                        .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut mesh.position.y,
                                            sim_bounds.min.y..=sim_bounds.max.y,
                                        )
                                        .text("Move Y")
                                        .suffix(" m")
                                        .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();
                                if !self.is_2d {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(
                                                &mut mesh.position.z,
                                                sim_bounds.min.z..=sim_bounds.max.z,
                                            )
                                            .text("Move Z")
                                            .suffix(" m")
                                            .clamping(egui::SliderClamping::Edits),
                                        )
                                        .changed();
                                } else {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(&mut mesh.position.z, -0.6..=0.6)
                                                .text("Slice Z")
                                                .suffix(" m")
                                                .clamping(egui::SliderClamping::Edits),
                                        )
                                        .changed();
                                }

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut mesh.scale, 0.04..=2.20)
                                            .text("Scale")
                                            .suffix(" m")
                                            .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();

                                let mut pitch_deg = mesh.rotation.x.to_degrees();
                                let mut yaw_deg = mesh.rotation.y.to_degrees();
                                let mut roll_deg = mesh.rotation.z.to_degrees();
                                let mut rot_changed = false;
                                rot_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut pitch_deg, -180.0..=180.0)
                                            .text("Rotate X")
                                            .suffix("°")
                                            .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();
                                rot_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut yaw_deg, -180.0..=180.0)
                                            .text("Rotate Y")
                                            .suffix("°")
                                            .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();
                                rot_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut roll_deg, -180.0..=180.0)
                                            .text("Rotate Z")
                                            .suffix("°")
                                            .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();

                                if rot_changed {
                                    mesh.rotation = Vec3::new(
                                        pitch_deg.to_radians(),
                                        yaw_deg.to_radians(),
                                        roll_deg.to_radians(),
                                    );
                                    changed = true;
                                }

                                if changed {
                                    mesh.update_transform();
                                }
                            }
                        }

                        egui::CollapsingHeader::new(format!(
                            "Surface Physics ({})",
                            collider.id
                        ))
                        .id_salt(("obs_surf_phys", collider.id))
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.add(
                                egui::Slider::new(&mut collider.restitution, 0.0..=0.8)
                                    .text("Bounce")
                                    .fixed_decimals(2),
                            );
                            ui.add(
                                egui::Slider::new(&mut collider.friction, 0.1..=1.0)
                                    .text("Slip")
                                    .fixed_decimals(2),
                            )
                            .on_hover_text("1.0 = frictionless slip, lower = higher surface drag");
                        });
                    }
                }
                if let Some(idx) = to_remove_col {
                    self.solver.colliders.remove(idx);
                }
            });

        let faucet_title = format!("Faucets & Emitters ({})", self.solver.faucets.len());
        egui::CollapsingHeader::new(faucet_title)
            .default_open(!self.solver.faucets.is_empty())
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("+ Waterfall").clicked() {
                        let id = self.next_faucet_id;
                        self.next_faucet_id += 1;
                        let mut f = Faucet::new_waterfall(
                            format!("Waterfall #{}", id),
                            Vec3::new(0.0, 1.1, 0.0),
                            0.4,
                            0.08,
                            600.0,
                        );
                        f.id = id;
                        self.solver.faucets.push(f);
                    }
                    if ui.button("+ Shower").clicked() {
                        let id = self.next_faucet_id;
                        self.next_faucet_id += 1;
                        let mut f = Faucet::new_shower(
                            format!("Shower #{}", id),
                            Vec3::new(0.0, 1.1, 0.0),
                            0.12,
                            500.0,
                            0.22,
                        );
                        f.id = id;
                        self.solver.faucets.push(f);
                    }
                    if ui.button("+ Tap").clicked() {
                        let id = self.next_faucet_id;
                        self.next_faucet_id += 1;
                        let mut f = Faucet::new_circle(
                            format!("Tap #{}", id),
                            Vec3::new(-0.5, 0.9, 0.0),
                            Vec3::new(1.8, -1.0, 0.0),
                            0.05,
                            400.0,
                        );
                        f.id = id;
                        self.solver.faucets.push(f);
                    }
                });

                let mut to_remove: Option<usize> = None;
                for (idx, faucet) in self.solver.faucets.iter_mut().enumerate() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut faucet.enabled, "");
                        ui.strong(&faucet.name);
                        if Self::trash_icon_button(ui, "Remove faucet / tap") {
                            to_remove = Some(idx);
                        }
                    });
                    if faucet.enabled {
                        ui.horizontal(|ui| {
                            ui.weak("Pos:");
                            ui.add(
                                egui::DragValue::new(&mut faucet.position.x)
                                    .speed(0.01)
                                    .prefix("X:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut faucet.position.y)
                                    .speed(0.01)
                                    .prefix("Y:"),
                            );
                            if !self.is_2d {
                                ui.add(
                                    egui::DragValue::new(&mut faucet.position.z)
                                        .speed(0.01)
                                        .prefix("Z:"),
                                );
                            } else {
                                faucet.position.z = 0.0;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.weak("Vel:");
                            ui.add(
                                egui::DragValue::new(&mut faucet.direction.x)
                                    .speed(0.05)
                                    .prefix("Vx:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut faucet.direction.y)
                                    .speed(0.05)
                                    .prefix("Vy:"),
                            );
                            if !self.is_2d {
                                ui.add(
                                    egui::DragValue::new(&mut faucet.direction.z)
                                        .speed(0.05)
                                        .prefix("Vz:"),
                                );
                            } else {
                                faucet.direction.z = 0.0;
                            }
                        });
                        match &mut faucet.shape {
                            FaucetShape::Circle { radius } => {
                                ui.add(
                                    egui::Slider::new(radius, 0.005..=0.40).text("Radius (m)"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut faucet.spray_spread, 0.0..=0.6)
                                        .text("Spray Angle"),
                                );
                            }
                            FaucetShape::Rectangle { width, depth } => {
                                ui.add(egui::Slider::new(width, 0.05..=1.4).text("Width (m)"));
                                if !self.is_2d {
                                    ui.add(
                                        egui::Slider::new(depth, 0.01..=0.6).text("Depth (m)"),
                                    );
                                }
                            }
                            FaucetShape::Point => {
                                let mut r = 0.0f32;
                                if ui
                                    .add(egui::Slider::new(&mut r, 0.0..=0.40).text("Radius (m)"))
                                    .changed()
                                    && r > 0.0
                                {
                                    faucet.shape = FaucetShape::Circle {
                                        radius: r.max(0.005),
                                    };
                                }
                            }
                        }
                        ui.add(
                            egui::Slider::new(&mut faucet.flow_rate, 25.0..=4000.0)
                                .text("Rate (p/s)"),
                        );
                    }
                }
                if let Some(idx) = to_remove {
                    self.solver.faucets.remove(idx);
                }
            });

        let drain_title = format!("Drains & Sinks ({})", self.solver.drains.len());
        egui::CollapsingHeader::new(drain_title)
            .default_open(!self.solver.drains.is_empty())
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("+ Floor Drain").clicked() {
                        let id = self.next_drain_id;
                        self.next_drain_id += 1;
                        let mut d = Drain::new(
                            format!("Floor Drain #{}", id),
                            CubeFace::Floor,
                            Vec3::ZERO,
                            0.18,
                        );
                        d.id = id;
                        self.solver.drains.push(d);
                    }
                    if ui.button("+ Wall Drain").clicked() {
                        let id = self.next_drain_id;
                        self.next_drain_id += 1;
                        let mut d = Drain::new(
                            format!("Wall Drain #{}", id),
                            CubeFace::Left,
                            Vec3::new(0.0, 0.2, 0.0),
                            0.15,
                        );
                        d.id = id;
                        self.solver.drains.push(d);
                    }
                    if ui.button("+ Point Sink").clicked() {
                        let id = self.next_drain_id;
                        self.next_drain_id += 1;
                        let mut d = Drain::new(
                            format!("Sink #{}", id),
                            CubeFace::Custom,
                            Vec3::new(0.0, 0.35, 0.0),
                            0.14,
                        );
                        d.id = id;
                        self.solver.drains.push(d);
                    }
                });

                let sim_bounds = self.solver.bounds;
                let mut to_remove: Option<usize> = None;
                for (idx, drain) in self.solver.drains.iter_mut().enumerate() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut drain.enabled, "");
                        ui.strong(&drain.name);
                        if Self::trash_icon_button(ui, "Remove drain") {
                            to_remove = Some(idx);
                        }
                    });
                    if drain.enabled {
                        egui::ComboBox::from_id_salt(("drain_face", drain.id))
                            .selected_text(drain.face.name())
                            .width(210.0)
                            .show_ui(ui, |ui| {
                                for &face in CubeFace::ALL {
                                    if self.is_2d
                                        && matches!(face, CubeFace::Back | CubeFace::Front)
                                    {
                                        continue;
                                    }
                                    ui.selectable_value(&mut drain.face, face, face.name());
                                }
                            });
                        ui.add(
                            egui::Slider::new(&mut drain.radius, 0.03..=0.60)
                                .text("Radius")
                                .suffix(" m"),
                        );
                        ui.horizontal(|ui| {
                            ui.weak("Offset:");
                            if !matches!(drain.face, CubeFace::Left | CubeFace::Right) {
                                ui.add(
                                    egui::DragValue::new(&mut drain.position.x)
                                        .speed(0.01)
                                        .range(sim_bounds.min.x..=sim_bounds.max.x)
                                        .prefix("X:"),
                                );
                            }
                            if !matches!(drain.face, CubeFace::Floor | CubeFace::Ceiling) {
                                ui.add(
                                    egui::DragValue::new(&mut drain.position.y)
                                        .speed(0.01)
                                        .range(sim_bounds.min.y..=sim_bounds.max.y)
                                        .prefix("Y:"),
                                );
                            }
                            if !self.is_2d
                                && !matches!(drain.face, CubeFace::Back | CubeFace::Front)
                            {
                                ui.add(
                                    egui::DragValue::new(&mut drain.position.z)
                                        .speed(0.01)
                                        .range(sim_bounds.min.z..=sim_bounds.max.z)
                                        .prefix("Z:"),
                                );
                            } else {
                                drain.position.z = 0.0;
                            }
                        });
                    }
                }
                if let Some(idx) = to_remove {
                    self.solver.drains.remove(idx);
                }
            });
    }

    fn draw_physics_tab(&mut self, ui: &mut egui::Ui) {
        // 1. Curated Physical Fluid Substance Presets
        Self::section_card(ui, "Fluid Substance Preset", |ui| {
            let mut chosen = self.fluid_preset;
            egui::ComboBox::from_id_salt("fluid_phys_preset")
                .selected_text(chosen.name())
                .width(268.0)
                .show_ui(ui, |ui| {
                    for &p in FluidPhysicsPreset::ALL {
                        if ui.selectable_value(&mut chosen, p, p.name()).clicked() {
                            self.apply_fluid_preset(p);
                        }
                    }
                });

            ui.horizontal_wrapped(|ui| {
                for &(p, lbl) in &[
                    (FluidPhysicsPreset::Water, "Water"),
                    (FluidPhysicsPreset::Ocean, "Ocean"),
                    (FluidPhysicsPreset::Oil, "Oil"),
                    (FluidPhysicsPreset::Honey, "Honey"),
                    (FluidPhysicsPreset::Mercury, "Mercury"),
                    (FluidPhysicsPreset::Lava, "Lava"),
                ] {
                    if ui
                        .selectable_label(self.fluid_preset == p, lbl)
                        .clicked()
                    {
                        self.apply_fluid_preset(p);
                    }
                }
            });

            if self.fluid_preset != FluidPhysicsPreset::Custom {
                if Self::icon_button(
                    ui,
                    UiIcon::Mesh,
                    "Match Visual Material & Colormap",
                    false,
                    Some(egui::vec2(ui.available_width(), 24.0)),
                )
                .on_hover_text("Automatically sync liquid optical material & particle colormap to match this substance")
                .clicked()
                {
                    self.apply_matching_visual_material();
                }
            }
        });

        ui.add_space(6.0);

        // 2. Primary Natural Fluid Dynamics Controls
        Self::section_card(ui, "Core Fluid Dynamics", |ui| {
            let mut edited = false;

            edited |= ui
                .add(
                    egui::Slider::new(&mut self.solver.config.gravity.y, -25.0..=5.0)
                        .text("Gravity Y")
                        .suffix(" m/s²"),
                )
                .changed();

            edited |= ui
                .add(
                    egui::Slider::new(&mut self.solver.config.viscosity, 0.0..=0.95)
                        .text("Viscosity")
                        .fixed_decimals(3),
                )
                .on_hover_text("XSPH shear viscosity: 0.03–0.08 crisp water, 0.35 oil, 0.72+ honey/lava")
                .changed();

            edited |= ui
                .add(
                    egui::Slider::new(&mut self.solver.config.surface_tension, 0.0..=0.30)
                        .text("Surface Tension")
                        .fixed_decimals(3),
                )
                .on_hover_text("Akinci free-surface cohesion: pulls airborne sheets into natural rounded droplets")
                .changed();

            edited |= ui
                .add(
                    egui::Slider::new(&mut self.solver.config.vorticity, 0.0..=0.20)
                        .text("Vorticity Swirl")
                        .fixed_decimals(3),
                )
                .on_hover_text("Macklin & Müller vorticity confinement: preserves energetic turbulent eddies while keeping calm pools still")
                .changed();

            edited |= ui
                .add(
                    egui::Slider::new(&mut self.solver.config.air_drag, 0.0..=2.0)
                        .text("Spray Air Drag")
                        .fixed_decimals(2),
                )
                .on_hover_text("Aerodynamic drag acting strictly on sparse airborne splash droplets so mist arcs naturally instead of flying ballistically")
                .changed();

            if edited {
                self.fluid_preset = FluidPhysicsPreset::Custom;
            }

            ui.separator();
            ui.add(
                egui::Slider::new(&mut self.sim_speed, 0.10..=2.00)
                    .text("Sim Speed")
                    .suffix("x")
                    .fixed_decimals(2),
            )
            .on_hover_text("Time-scale multiplier (e.g. 0.25x for cinematic slow-motion splashes)");

            if Self::icon_button(ui, UiIcon::Reset, "Restore Pure Water Defaults", false, None)
                .clicked()
            {
                self.apply_fluid_preset(FluidPhysicsPreset::Water);
                self.solver.config.gravity = Vec3::new(0.0, -9.81, 0.0);
                self.sim_speed = 1.0;
            }
        });

        ui.add_space(6.0);

        // 3. Collapsed Advanced Solver, Lateral Gravity, Boundary & Compute Backend
        egui::CollapsingHeader::new("Advanced Solver, Gravity Tilt & Compute")
            .default_open(false)
            .show(ui, |ui| {
                Self::section_card(ui, "Solver & Boundary Parameters", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.solver.config.substeps, 1..=10)
                            .text("Substeps / Frame"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.solver.config.solver_iterations, 1..=10)
                            .text("Pressure Iters"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.solver.config.tensile_k, 0.0..=0.012)
                            .text("Tensile Anti-Clump")
                            .fixed_decimals(4),
                    )
                    .on_hover_text("Monaghan artificial pressure preventing particle pairing at free surfaces");
                    ui.add(
                        egui::Slider::new(&mut self.solver.config.max_velocity, 3.0..=40.0)
                            .text("Max Speed")
                            .suffix(" m/s"),
                    );

                    ui.separator();
                    ui.label("Lateral Gravity Tilt:");
                    ui.add(
                        egui::Slider::new(&mut self.solver.config.gravity.x, -12.0..=12.0)
                            .text("Gravity X")
                            .suffix(" m/s²"),
                    );
                    if !self.is_2d {
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.gravity.z, -12.0..=12.0)
                                .text("Gravity Z")
                                .suffix(" m/s²"),
                        );
                    }

                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut self.solver.config.boundary_friction, 0.2..=1.0)
                            .text("Wall Slip")
                            .fixed_decimals(2),
                    )
                    .on_hover_text("1.0 = free-slip glass walls, lower = viscous wall drag");
                    ui.add(
                        egui::Slider::new(&mut self.solver.config.restitution, 0.0..=0.4)
                            .text("Wall Bounce")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.solver.mouse_force_strength, 5.0..=140.0)
                            .text("Cursor Force"),
                    );

                    ui.separator();
                    if self.gpu_solver.is_some() {
                        let prev = self.use_gpu;
                        ui.checkbox(&mut self.use_gpu, "Vulkan GPU Compute Acceleration");
                        if !prev && self.use_gpu {
                            if let Some(ref mut gpu) = self.gpu_solver {
                                gpu.sync_particles(
                                    self.solver.bounds,
                                    &self.solver.config,
                                    &self.solver.particles,
                                    false,
                                );
                            }
                        }
                        ui.weak(format!("Adapter: {}", self.gpu_device_name));
                    } else {
                        ui.weak("Vulkan GPU unavailable — using multi-core Rayon CPU solver.");
                    }
                });
            });
    }

    fn draw_surface_tab(&mut self, ui: &mut egui::Ui) {
        // 1. Primary Render Mode Selector (Particles vs Surface Mesh)
        Self::section_card(ui, "Fluid Representation", |ui| {
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 6.0) * 0.5;
                if Self::icon_button(
                    ui,
                    UiIcon::Spheres,
                    "SPH Spheres",
                    self.render_mode == RenderMode::Particles,
                    Some(egui::vec2(w, 28.0)),
                )
                .clicked()
                {
                    self.render_mode = RenderMode::Particles;
                }
                if Self::icon_button(
                    ui,
                    UiIcon::Mesh,
                    "3D Surface Mesh",
                    self.render_mode == RenderMode::SurfaceMesh,
                    Some(egui::vec2(w, 28.0)),
                )
                .clicked()
                {
                    self.render_mode = RenderMode::SurfaceMesh;
                    self.mesh_dirty = true;
                }
            });
            ui.weak(match self.render_mode {
                RenderMode::Particles => {
                    "Instanced analytical ray-sphere impostors with Fresnel rim lighting & whitewater foam."
                }
                RenderMode::SurfaceMesh => {
                    "Continuous Marching Cubes isosurface with Beer-Lambert absorption, textures & PBR light."
                }
            });
        });

        ui.add_space(6.0);

        if self.render_mode == RenderMode::Particles {
            Self::section_card(ui, "Particle Sphere Shading & Colormap", |ui| {
                ui.label("Color Field:");
                egui::ComboBox::from_id_salt("p_color_mode")
                    .selected_text(self.color_mode.name())
                    .width(268.0)
                    .show_ui(ui, |ui| {
                        for &mode in ColorMode::ALL {
                            ui.selectable_value(&mut self.color_mode, mode, mode.name());
                        }
                    });

                ui.label("Colormap Gradient:");
                egui::ComboBox::from_id_salt("p_color_map")
                    .selected_text(self.color_map.name())
                    .width(268.0)
                    .show_ui(ui, |ui| {
                        for &cm in ColorMapType::ALL {
                            ui.selectable_value(&mut self.color_map, cm, cm.name());
                        }
                    });

                ui.horizontal_wrapped(|ui| {
                    for &cm in ColorMapType::ALL {
                        if ui
                            .selectable_label(self.color_map == cm, cm.name())
                            .clicked()
                        {
                            self.color_map = cm;
                        }
                    }
                });

                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.display_size_factor, 0.5..=2.0)
                        .text("Sphere Scale")
                        .fixed_decimals(2),
                );
                ui.checkbox(&mut self.lit_spheres, "Glossy Fresnel & Specular Shading");
                ui.checkbox(
                    &mut self.particle_foam,
                    "Dynamic Aerated Whitewater Foam",
                )
                .on_hover_text("Highlights fast, low-density crest & splash droplets as bright whitewater foam");
                ui.checkbox(&mut self.enable_culling, "Frustum Viewport Culling");
            });
        } else {
            // === Surface Mesh Controls (Clean Primary + Collapsed Advanced Drawers) ===
            Self::section_card(ui, "Material Preset & Liquid Color", |ui| {
                let mut chosen_preset = self.mesh_config.preset;
                egui::ComboBox::from_id_salt("mesh_mat_preset")
                    .selected_text(chosen_preset.name())
                    .width(268.0)
                    .show_ui(ui, |ui| {
                        for &p in FluidMaterialPreset::ALL {
                            if ui
                                .selectable_value(&mut chosen_preset, p, p.name())
                                .clicked()
                            {
                                p.apply(&mut self.mesh_config);
                            }
                        }
                    });

                ui.add_space(3.0);
                ui.horizontal(|ui| {
                    ui.label("Shallow / Surface:");
                    if ui
                        .color_edit_button_rgb(&mut self.mesh_config.shallow_color)
                        .changed()
                    {
                        self.mesh_config.preset = FluidMaterialPreset::Custom;
                    }
                    ui.monospace(SurfaceMeshConfig::format_hex(self.mesh_config.shallow_color));
                });

                ui.horizontal(|ui| {
                    ui.label("Deep Absorption:  ");
                    if ui
                        .color_edit_button_rgb(&mut self.mesh_config.deep_color)
                        .changed()
                    {
                        self.mesh_config.preset = FluidMaterialPreset::Custom;
                    }
                    ui.monospace(SurfaceMeshConfig::format_hex(self.mesh_config.deep_color));
                });

                if ui
                    .add(
                        egui::Slider::new(&mut self.mesh_config.opacity, 0.08..=1.0)
                            .text("Surface Opacity"),
                    )
                    .changed()
                {
                    self.mesh_config.preset = FluidMaterialPreset::Custom;
                }
                if ui
                    .add(
                        egui::Slider::new(&mut self.mesh_config.absorption, 0.1..=7.0)
                            .text("Volume Absorption"),
                    )
                    .changed()
                {
                    self.mesh_config.preset = FluidMaterialPreset::Custom;
                }
                ui.add(
                    egui::Slider::new(&mut self.mesh_config.foam_intensity, 0.0..=1.0)
                        .text("Crest Foam"),
                );
            });

            ui.add_space(6.0);

            Self::section_card(ui, "Procedural Surface Texture", |ui| {
                egui::ComboBox::from_id_salt("mesh_tex_mode")
                    .selected_text(self.mesh_config.texture_mode.name())
                    .width(268.0)
                    .show_ui(ui, |ui| {
                        for &tm in MeshTextureMode::ALL {
                            ui.selectable_value(&mut self.mesh_config.texture_mode, tm, tm.name());
                        }
                    });

                if self.mesh_config.texture_mode != MeshTextureMode::None {
                    ui.add(
                        egui::Slider::new(&mut self.mesh_config.texture_scale, 0.8..=20.0)
                            .text("Texture Scale"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.mesh_config.texture_strength, 0.0..=1.0)
                            .text("Bump / Intensity"),
                    );
                }
            });

            ui.add_space(6.0);

            egui::CollapsingHeader::new("Advanced PBR Optics & Refraction")
                .default_open(false)
                .show(ui, |ui| {
                    Self::section_card(ui, "Dielectric & Metallic Optics", |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("IOR:");
                            for &(ior_val, label) in &[
                                (1.333f32, "Water 1.33"),
                                (1.470f32, "Glycerin 1.47"),
                                (1.520f32, "Glass 1.52"),
                                (2.400f32, "Chrome 2.4"),
                            ] {
                                if ui
                                    .selectable_label(
                                        (self.mesh_config.ior - ior_val).abs() < 0.02,
                                        label,
                                    )
                                    .clicked()
                                {
                                    self.mesh_config.ior = ior_val;
                                }
                            }
                        });

                        ui.add(
                            egui::Slider::new(&mut self.mesh_config.ior, 1.02..=2.60)
                                .text("Fresnel IOR")
                                .fixed_decimals(3),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.mesh_config.roughness, 0.015..=0.80)
                                .text("GGX Roughness")
                                .fixed_decimals(3),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.mesh_config.specular_intensity, 0.0..=3.0)
                                .text("Sun Specular"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.mesh_config.refraction_strength, 0.0..=1.2)
                                .text("Refraction"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.mesh_config.subsurface_scattering,
                                0.0..=1.6,
                            )
                            .text("Subsurface Glow"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.mesh_config.env_reflection, 0.0..=2.2)
                                .text("Sky Reflection"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.mesh_config.metallic, 0.0..=1.0)
                                .text("Metallic"),
                        );
                        ui.checkbox(
                            &mut self.mesh_config.cast_shadows,
                            "Cast Colored Shadows & Caustics (MP4 / PNG)",
                        );
                    });
                });

            egui::CollapsingHeader::new("Advanced Isosurface Meshing Grid")
                .default_open(false)
                .show(ui, |ui| {
                    Self::section_card(ui, "Marching Cubes Reconstruction", |ui| {
                        let mut geom_changed = false;
                        geom_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.mesh_config.voxel_scale, 0.40..=1.35)
                                    .text("Voxel Scale"),
                            )
                            .on_hover_text("Smaller = finer surface triangles; larger = faster preview")
                            .changed();
                        geom_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.mesh_config.iso_value, 0.16..=0.75)
                                    .text("Iso Threshold"),
                            )
                            .changed();
                        geom_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.mesh_config.smoothing_passes, 0..=10)
                                    .text("Taubin Smooth"),
                            )
                            .changed();
                        geom_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.mesh_config.kernel_scale, 0.7..=1.6)
                                    .text("Kernel Support"),
                            )
                            .changed();

                        if geom_changed {
                            self.mesh_dirty = true;
                        }

                        ui.horizontal(|ui| {
                            ui.weak(format!(
                                "Mesh: {} tris ({:.1} ms)",
                                self.cached_surface_mesh.triangle_count(),
                                self.last_mesh_time_ms
                            ));
                            if Self::icon_button(ui, UiIcon::Save, "Export .OBJ", false, None)
                                .clicked()
                            {
                                self.export_current_surface_obj();
                            }
                        });
                    });
                });
        }

        ui.add_space(6.0);

        // Shared Studio Backdrop, Sun Lighting & Camera Turntable (applies to BOTH Spheres & Mesh!)
        Self::section_card(ui, "Studio Environment, Sun & Camera", |ui| {
            ui.label("Sky & Backdrop:");
            egui::ComboBox::from_id_salt("studio_env_preset")
                .selected_text(self.mesh_config.env_preset.name())
                .width(268.0)
                .show_ui(ui, |ui| {
                    for &ep in EnvironmentPreset::ALL {
                        ui.selectable_value(&mut self.mesh_config.env_preset, ep, ep.name());
                    }
                });

            ui.add(
                egui::Slider::new(&mut self.mesh_config.exposure, 0.35..=2.40)
                    .text("Exposure")
                    .fixed_decimals(2),
            );

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.mesh_config.show_grid, "Floor Grid");
                ui.checkbox(&mut self.mesh_config.show_wireframe, "Chamber Wireframe");
            });

            let mut sun_changed = false;
            sun_changed |= ui
                .add(
                    egui::Slider::new(&mut self.sun_azimuth_deg, -180.0..=180.0)
                        .text("Sun Azimuth")
                        .suffix("°"),
                )
                .changed();
            sun_changed |= ui
                .add(
                    egui::Slider::new(&mut self.sun_elevation_deg, 8.0..=88.0)
                        .text("Sun Elevation")
                        .suffix("°"),
                )
                .changed();
            if sun_changed {
                self.update_dir_from_sun_angles();
            }

            egui::CollapsingHeader::new("Advanced Sun Radiance & Camera Lens")
                .default_open(false)
                .show(ui, |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.mesh_config.light_intensity, 0.1..=3.5)
                            .text("Sun Radiance"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.mesh_config.ambient_intensity, 0.05..=1.2)
                            .text("Ambient Fill"),
                    );

                    ui.separator();
                    let mut fov_deg = self.camera.fov_y.to_degrees();
                    if ui
                        .add(
                            egui::Slider::new(&mut fov_deg, 20.0..=90.0)
                                .text("Camera FOV")
                                .suffix("°"),
                        )
                        .changed()
                    {
                        self.camera.fov_y = fov_deg.to_radians();
                    }

                    if !self.is_2d {
                        ui.add(
                            egui::Slider::new(&mut self.camera_orbit_speed_deg, -60.0..=60.0)
                                .text("Turntable Orbit")
                                .suffix(" °/s"),
                        )
                        .on_hover_text("Automatic 360° turntable camera rotation in viewport & MP4 renders");
                        ui.add(
                            egui::Slider::new(&mut self.camera.distance, 0.8..=12.0)
                                .text("Cam Distance")
                                .suffix(" m"),
                        );
                    }
                    if Self::icon_button(ui, UiIcon::Camera, "Reset Camera", false, None).clicked()
                    {
                        self.reset_camera_framing();
                    }
                });
        });
    }

    fn format_eta(done: usize, total: usize, elapsed: f32) -> Option<String> {
        if done == 0 || total <= done || elapsed < 0.25 {
            return None;
        }
        let rem = (elapsed / done as f32) * (total - done) as f32;
        if !rem.is_finite() || rem < 0.4 {
            return Some("almost done".to_string());
        }
        let secs = rem.ceil() as u64;
        if secs >= 60 {
            Some(format!("~{}m {:02}s left", secs / 60, secs % 60))
        } else {
            Some(format!("~{}s left", secs))
        }
    }

    fn friendly_sim_quality_label(
        q: Option<SimulationQuality>,
        is_2d: bool,
        editor_q: SimulationQuality,
    ) -> String {
        match q {
            None => {
                let ed = match editor_q {
                    SimulationQuality::Draft => "Fast Draft",
                    SimulationQuality::Medium => "Balanced",
                    SimulationQuality::High => "Smooth",
                    SimulationQuality::Ultra => "High Detail",
                    SimulationQuality::Cinematic => "Cinematic",
                    SimulationQuality::Extreme => "Ultra VFX",
                    SimulationQuality::Studio => "Studio",
                    SimulationQuality::Film => "Film",
                    SimulationQuality::Master => "Master",
                    SimulationQuality::Custom => "Custom",
                };
                format!("Same as 3D Editor ({})", ed)
            }
            Some(SimulationQuality::Draft) => {
                if is_2d {
                    "Fast Draft (~300 particles)".to_string()
                } else {
                    "Fast Draft (~2.5k particles)".to_string()
                }
            }
            Some(SimulationQuality::Medium) => {
                if is_2d {
                    "Balanced (~500 particles)".to_string()
                } else {
                    "Balanced (~5.5k particles)".to_string()
                }
            }
            Some(SimulationQuality::High) => {
                if is_2d {
                    "Smooth (~1.5k particles)".to_string()
                } else {
                    "Smooth (~18k particles)".to_string()
                }
            }
            Some(SimulationQuality::Ultra) => {
                if is_2d {
                    "High Detail (~4k — Recommended)".to_string()
                } else {
                    "High Detail (~50k — Recommended)".to_string()
                }
            }
            Some(SimulationQuality::Cinematic) => {
                if is_2d {
                    "Cinematic (~10k particles)".to_string()
                } else {
                    "Cinematic (~120k particles)".to_string()
                }
            }
            Some(SimulationQuality::Extreme) => {
                if is_2d {
                    "Ultra VFX (~30k particles)".to_string()
                } else {
                    "Ultra VFX (~1 Million particles)".to_string()
                }
            }
            Some(SimulationQuality::Studio) => {
                if is_2d {
                    "Studio Offline (~48k particles)".to_string()
                } else {
                    "Studio Offline (~2.5 Million particles)".to_string()
                }
            }
            Some(SimulationQuality::Film) => {
                if is_2d {
                    "Film Offline (~95k particles)".to_string()
                } else {
                    "Film Offline (~4.9 Million particles)".to_string()
                }
            }
            Some(SimulationQuality::Master) => {
                if is_2d {
                    "Master Offline (~160k particles)".to_string()
                } else {
                    "Master Offline (~8.2 Million particles)".to_string()
                }
            }
            Some(SimulationQuality::Custom) => "Custom Setup".to_string(),
        }
    }

    fn draw_render_tab(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let (
            is_running,
            bg_kind,
            done,
            total,
            elapsed,
            phase_lbl,
            sim_pts,
            has_still,
            anim_len,
            backend_lbl,
            out_w,
            out_h,
            qual,
            saved_path,
        ) = {
            if let Ok(g) = self.bg_render.lock() {
                (
                    g.is_running,
                    g.kind,
                    g.completed_frames,
                    g.total_frames,
                    g.elapsed_secs,
                    g.phase_label.clone(),
                    g.sim_particles_count,
                    g.full_res_frame.is_some(),
                    g.anim_frames.len(),
                    g.backend_label.clone(),
                    g.output_width,
                    g.output_height,
                    g.quality,
                    g.saved_path.clone(),
                )
            } else {
                (
                    false,
                    StudioRenderKind::StillFrame,
                    0,
                    0,
                    0.0,
                    String::new(),
                    0,
                    false,
                    0,
                    String::new(),
                    self.video_width,
                    self.video_height,
                    self.gui_render_quality,
                    None,
                )
            }
        };
        let has_result = has_still || anim_len > 0;

        // 1. Beginner-Friendly One-Click Render Presets
        Self::section_card(ui, "1. Quick Render Presets", |ui| {
            let half_w = (ui.available_width() - 6.0) * 0.5;
            ui.horizontal(|ui| {
                for &p in &[
                    GuidedRenderPreset::QuickPreview,
                    GuidedRenderPreset::RecommendedHd,
                ] {
                    let ic = if p == GuidedRenderPreset::QuickPreview {
                        UiIcon::Bolt
                    } else {
                        UiIcon::Film
                    };
                    if Self::icon_button(
                        ui,
                        ic,
                        p.title(),
                        self.guided_render_preset == p,
                        Some(egui::vec2(half_w, 27.0)),
                    )
                    .on_hover_text(p.summary())
                    .clicked()
                    {
                        self.apply_guided_render_preset(p);
                    }
                }
            });
            ui.horizontal(|ui| {
                for &p in &[
                    GuidedRenderPreset::Cinematic2K,
                    GuidedRenderPreset::Ultra4K,
                ] {
                    if Self::icon_button(
                        ui,
                        UiIcon::Film,
                        p.title(),
                        self.guided_render_preset == p,
                        Some(egui::vec2(half_w, 27.0)),
                    )
                    .on_hover_text(p.summary())
                    .clicked()
                    {
                        self.apply_guided_render_preset(p);
                    }
                }
            });
            if Self::icon_button(
                ui,
                UiIcon::Camera,
                GuidedRenderPreset::StudioPhoto.title(),
                self.guided_render_preset == GuidedRenderPreset::StudioPhoto,
                Some(egui::vec2(ui.available_width(), 26.0)),
            )
            .on_hover_text(GuidedRenderPreset::StudioPhoto.summary())
            .clicked()
            {
                self.apply_guided_render_preset(GuidedRenderPreset::StudioPhoto);
            }

            ui.add_space(2.0);
            ui.colored_label(
                egui::Color32::from_rgb(148, 194, 236),
                self.guided_render_preset.summary(),
            );
        });

        ui.add_space(6.0);

        // 2. Plain-English Quality, Look & File Settings
        Self::section_card(ui, "2. Customize Look & Quality", |ui| {
            // Output Type: Video vs Still Photo
            ui.horizontal(|ui| {
                let half_w = (ui.available_width() - 6.0) * 0.5;
                if Self::icon_button(
                    ui,
                    UiIcon::Film,
                    "Video (.mp4)",
                    self.render_kind == StudioRenderKind::Animation,
                    Some(egui::vec2(half_w, 27.0)),
                )
                .on_hover_text("Simulate and render a smooth video animation you can watch & save")
                .clicked()
                {
                    self.render_kind = StudioRenderKind::Animation;
                    self.guided_render_preset = GuidedRenderPreset::Custom;
                }
                if Self::icon_button(
                    ui,
                    UiIcon::Camera,
                    "Still Photo (.png)",
                    self.render_kind == StudioRenderKind::StillFrame,
                    Some(egui::vec2(half_w, 27.0)),
                )
                .on_hover_text("Render a high-resolution studio photograph")
                .clicked()
                {
                    self.render_kind = StudioRenderKind::StillFrame;
                    self.guided_render_preset = GuidedRenderPreset::Custom;
                }
            });

            ui.add_space(3.0);

            // Liquid Appearance (decoupled from Live 3D Editor so viewport stays fast!)
            ui.label("Liquid Style in Render:");
            ui.horizontal(|ui| {
                let third_w = (ui.available_width() - 12.0) / 3.0;
                if Self::icon_button(
                    ui,
                    UiIcon::Mesh,
                    "Smooth Liquid",
                    self.render_appearance == RenderAppearance::SmoothLiquid,
                    Some(egui::vec2(third_w + 8.0, 25.0)),
                )
                .on_hover_text("Render realistic glossy 3D liquid surface (keeps your live editor fast)")
                .clicked()
                {
                    self.render_appearance = RenderAppearance::SmoothLiquid;
                    self.guided_render_preset = GuidedRenderPreset::Custom;
                }
                if Self::icon_button(
                    ui,
                    UiIcon::Spheres,
                    "Droplets",
                    self.render_appearance == RenderAppearance::ParticleSpheres,
                    Some(egui::vec2(third_w - 4.0, 25.0)),
                )
                .on_hover_text("Render individual glossy fluid spheres")
                .clicked()
                {
                    self.render_appearance = RenderAppearance::ParticleSpheres;
                    self.guided_render_preset = GuidedRenderPreset::Custom;
                }
                if Self::icon_button(
                    ui,
                    UiIcon::Cube3D,
                    "Same as View",
                    self.render_appearance == RenderAppearance::MatchViewport,
                    Some(egui::vec2(third_w - 4.0, 25.0)),
                )
                .on_hover_text("Use whichever mode (Spheres or Mesh) is active in the top bar")
                .clicked()
                {
                    self.render_appearance = RenderAppearance::MatchViewport;
                    self.guided_render_preset = GuidedRenderPreset::Custom;
                }
            });

            // Quick Material & Sky Lighting selectors right in place
            egui::Grid::new("render_tab_look_grid")
                .num_columns(2)
                .spacing([8.0, 5.0])
                .show(ui, |ui| {
                    if self.effective_render_mode() == RenderMode::SurfaceMesh {
                        ui.label("Material:");
                        let mut mat = self.mesh_config.preset;
                        egui::ComboBox::from_id_salt("render_tab_mat_combo")
                            .selected_text(mat.name())
                            .width(215.0)
                            .show_ui(ui, |ui| {
                                for &p in FluidMaterialPreset::ALL {
                                    if ui.selectable_value(&mut mat, p, p.name()).clicked() {
                                        p.apply(&mut self.mesh_config);
                                    }
                                }
                            });
                        ui.end_row();
                    } else {
                        ui.label("Color Theme:");
                        egui::ComboBox::from_id_salt("render_tab_cmap_combo")
                            .selected_text(self.color_map.name())
                            .width(215.0)
                            .show_ui(ui, |ui| {
                                for &cm in ColorMapType::ALL {
                                    ui.selectable_value(&mut self.color_map, cm, cm.name());
                                }
                            });
                        ui.end_row();
                    }

                    ui.label("Sky Lighting:");
                    egui::ComboBox::from_id_salt("render_tab_env_combo")
                        .selected_text(self.mesh_config.env_preset.name())
                        .width(215.0)
                        .show_ui(ui, |ui| {
                            for &ep in EnvironmentPreset::ALL {
                                ui.selectable_value(
                                    &mut self.mesh_config.env_preset,
                                    ep,
                                    ep.name(),
                                );
                            }
                        });
                    ui.end_row();

                    ui.label("Fluid Detail:");
                    let cur_q_lbl = Self::friendly_sim_quality_label(
                        self.render_sim_quality,
                        self.is_2d,
                        self.current_quality,
                    );
                    egui::ComboBox::from_id_salt("render_tab_sim_qual")
                        .selected_text(cur_q_lbl)
                        .width(215.0)
                        .show_ui(ui, |ui| {
                            let same_lbl = Self::friendly_sim_quality_label(
                                None,
                                self.is_2d,
                                self.current_quality,
                            );
                            if ui
                                .selectable_value(&mut self.render_sim_quality, None, same_lbl)
                                .clicked()
                            {
                                self.guided_render_preset = GuidedRenderPreset::Custom;
                            }
                            ui.separator();
                            for &q in SimulationQuality::ALL
                                .iter()
                                .chain(SimulationQuality::CLI_ONLY.iter())
                            {
                                let lbl = Self::friendly_sim_quality_label(
                                    Some(q),
                                    self.is_2d,
                                    self.current_quality,
                                );
                                if ui
                                    .selectable_value(&mut self.render_sim_quality, Some(q), lbl)
                                    .clicked()
                                {
                                    self.guided_render_preset = GuidedRenderPreset::Custom;
                                }
                            }
                        });
                    ui.end_row();
                });

            if !self.is_2d
                && matches!(
                    self.render_sim_quality,
                    Some(
                        SimulationQuality::Extreme
                            | SimulationQuality::Studio
                            | SimulationQuality::Film
                            | SimulationQuality::Master
                    )
                )
            {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 196, 92),
                    "Note: 1M+ particle VFX tiers run offline in the background without freezing the editor, but require high RAM/VRAM and take longer.",
                );
            }

            ui.separator();

            // Picture Size
            ui.horizontal_wrapped(|ui| {
                ui.label("Picture Size:");
                for &(w, h, lbl) in &[
                    (1280usize, 720usize, "720p"),
                    (1920usize, 1080usize, "1080p HD"),
                    (2560usize, 1440usize, "2K"),
                    (3840usize, 2160usize, "4K"),
                    (1080usize, 1920usize, "9:16 Phone"),
                ] {
                    if ui
                        .selectable_label(self.video_width == w && self.video_height == h, lbl)
                        .clicked()
                    {
                        self.video_width = w;
                        self.video_height = h;
                        self.guided_render_preset = GuidedRenderPreset::Custom;
                    }
                }
            });

            // Image Sharpness & Shadows (friendly names for CliRenderQuality)
            ui.horizontal_wrapped(|ui| {
                ui.label("Sharpness:");
                for &(rq, lbl, tip) in &[
                    (
                        CliRenderQuality::Standard,
                        "Fast (1x)",
                        "Fastest render speed with standard shadows",
                    ),
                    (
                        CliRenderQuality::Studio,
                        "Smooth (2x)",
                        "Recommended: 2x anti-aliasing & soft studio shadows",
                    ),
                    (
                        CliRenderQuality::Cinema,
                        "Crisp (3x)",
                        "Cinema-grade 3x anti-aliasing & ultra-sharp edges",
                    ),
                    (
                        CliRenderQuality::Master,
                        "Max (4x)",
                        "Maximum 4x supersampling (slowest, finest detail)",
                    ),
                ] {
                    if ui
                        .selectable_label(self.gui_render_quality == rq, lbl)
                        .on_hover_text(tip)
                        .clicked()
                    {
                        self.gui_render_quality = rq;
                        self.guided_render_preset = GuidedRenderPreset::Custom;
                    }
                }
            });

            ui.separator();

            // Mode-specific simple controls (Video Timeline vs Still Photo Capture)
            match self.render_kind {
                StudioRenderKind::Animation => {
                    if ui
                        .add(
                            egui::Slider::new(&mut self.cli_duration, 1.0..=30.0)
                                .text("Video Length")
                                .suffix(" sec")
                                .fixed_decimals(1),
                        )
                        .changed()
                    {
                        self.guided_render_preset = GuidedRenderPreset::Custom;
                    }

                    let planned_frames =
                        (self.cli_duration * self.cli_fps.max(1) as f32).round().max(1.0) as usize;
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Smoothness:");
                        for &(f, lbl) in &[
                            (24u32, "24 FPS"),
                            (30u32, "30 FPS"),
                            (60u32, "60 FPS"),
                        ] {
                            if ui.selectable_label(self.cli_fps == f, lbl).clicked() {
                                self.cli_fps = f;
                                self.guided_render_preset = GuidedRenderPreset::Custom;
                            }
                        }
                        ui.weak(format!("({} frames)", planned_frames));
                    });

                    ui.horizontal_wrapped(|ui| {
                        ui.label("Motion:");
                        for &(spd, lbl) in &[
                            (1.0f32, "Normal"),
                            (0.5f32, "Slow-Mo"),
                            (0.25f32, "Super Slow"),
                        ] {
                            if ui
                                .selectable_label((self.sim_speed - spd).abs() < 0.03, lbl)
                                .clicked()
                            {
                                self.sim_speed = spd;
                            }
                        }
                    });

                    if !self.is_2d {
                        let mut spin = self.camera_orbit_speed_deg.abs() > 0.5;
                        if ui
                            .checkbox(&mut spin, "360° Showcase Camera Spin")
                            .on_hover_text(
                                "Gently orbits the camera around the tank while the video renders",
                            )
                            .changed()
                        {
                            self.camera_orbit_speed_deg = if spin { 18.0 } else { 0.0 };
                        }
                    }

                    ui.checkbox(&mut self.render_save_mp4, "Save video file (.mp4) to disk");
                    if self.render_save_mp4 {
                        ui.horizontal(|ui| {
                            let btn_w = 78.0;
                            let txt_w = (ui.available_width() - btn_w - 6.0).max(80.0);
                            ui.add_sized(
                                [txt_w, 22.0],
                                egui::TextEdit::singleline(&mut self.cli_mp4_output),
                            );
                            if Self::icon_button(
                                ui,
                                UiIcon::Folder,
                                "Browse",
                                false,
                                Some(egui::vec2(btn_w, 22.0)),
                            )
                            .on_hover_text("Choose where to save your .mp4 video")
                            .clicked()
                            {
                                self.spawn_render_file_dialog(ctx.clone(), 0);
                            }
                        });
                    }
                }
                StudioRenderKind::StillFrame => {
                    ui.horizontal(|ui| {
                        ui.label("Moment:");
                        if ui
                            .selectable_label(!self.render_still_simulate, "Current View")
                            .on_hover_text("Capture the current moment shown in the 3D editor")
                            .clicked()
                        {
                            self.render_still_simulate = false;
                        }
                        if ui
                            .selectable_label(self.render_still_simulate, "Simulate to Time...")
                            .on_hover_text(
                                "Simulate from the start (t = 0s) up to an exact timestamp before taking the photo",
                            )
                            .clicked()
                        {
                            self.render_still_simulate = true;
                            if self.render_still_time < 0.05 {
                                self.render_still_time =
                                    if self.sim_time > 0.05 { self.sim_time } else { 1.6 };
                            }
                        }
                    });

                    if self.render_still_simulate {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Slider::new(&mut self.render_still_time, 0.0..=15.0)
                                    .text("Time")
                                    .suffix(" sec")
                                    .fixed_decimals(2),
                            );
                            if ui
                                .small_button("Use Editor")
                                .on_hover_text("Copy current simulation timestamp from the live editor")
                                .clicked()
                            {
                                self.render_still_time = self.sim_time;
                            }
                        });
                    }

                    ui.checkbox(
                        &mut self.render_auto_save_png,
                        "Auto-save photo (.png) to disk",
                    );
                    if self.render_auto_save_png {
                        ui.horizontal(|ui| {
                            let btn_w = 78.0;
                            let txt_w = (ui.available_width() - btn_w - 6.0).max(80.0);
                            ui.add_sized(
                                [txt_w, 22.0],
                                egui::TextEdit::singleline(&mut self.png_output_path),
                            );
                            if Self::icon_button(
                                ui,
                                UiIcon::Folder,
                                "Browse",
                                false,
                                Some(egui::vec2(btn_w, 22.0)),
                            )
                            .on_hover_text("Choose where to save your .png photo")
                            .clicked()
                            {
                                self.spawn_render_file_dialog(ctx.clone(), 1);
                            }
                        });
                    }
                }
            }
        });

        ui.add_space(6.0);

        // 3. Primary Start Button, Plain-English Progress + ETA & End Result Actions
        Self::section_card(ui, "3. Start Render & View Result", |ui| {
            let look_name = match self.effective_render_mode() {
                RenderMode::SurfaceMesh => "Smooth Liquid",
                RenderMode::Particles => "Droplets",
            };

            if is_running {
                let frac = if total > 0 {
                    (done as f32 / total as f32).clamp(0.0, 1.0)
                } else {
                    0.25
                };
                let pct = (frac * 100.0).round() as u32;
                let bar_txt = if let Some(eta) = Self::format_eta(done, total, elapsed) {
                    format!("{}% • {}", pct, eta)
                } else if total > 1 {
                    format!("{}% ({}/{})", pct, done, total)
                } else {
                    "Working...".to_string()
                };

                if !phase_lbl.is_empty() {
                    ui.colored_label(egui::Color32::from_rgb(154, 214, 255), &phase_lbl);
                }
                ui.add(egui::ProgressBar::new(frac).animate(true).text(bar_txt));

                if sim_pts > 0 || !backend_lbl.is_empty() {
                    let pts_str = if sim_pts > 0 {
                        format!("{} particles", Self::format_particle_count(sim_pts))
                    } else {
                        String::new()
                    };
                    let info = match (!pts_str.is_empty(), !backend_lbl.is_empty()) {
                        (true, true) => format!("{} • {}", pts_str, backend_lbl),
                        (true, false) => pts_str,
                        (false, true) => backend_lbl.clone(),
                        (false, false) => String::new(),
                    };
                    ui.weak(info);
                }

                let stop_lbl = if bg_kind == StudioRenderKind::Animation {
                    "Stop Early & Keep Rendered Video"
                } else {
                    "Cancel Photo Render"
                };
                if Self::icon_button(
                    ui,
                    UiIcon::Pause,
                    stop_lbl,
                    false,
                    Some(egui::vec2(ui.available_width(), 27.0)),
                )
                .clicked()
                {
                    self.bg_render_cancel.store(true, Ordering::SeqCst);
                }
                ctx.request_repaint();
            } else {
                let summary_line = match self.render_kind {
                    StudioRenderKind::Animation => {
                        let frames = (self.cli_duration * self.cli_fps.max(1) as f32)
                            .round()
                            .max(1.0) as usize;
                        format!(
                            "Ready: {}×{} video • {:.1}s @ {} FPS ({} frames) • {}",
                            self.video_width,
                            self.video_height,
                            self.cli_duration,
                            self.cli_fps,
                            frames,
                            look_name
                        )
                    }
                    StudioRenderKind::StillFrame => {
                        format!(
                            "Ready: {}×{} photo • {} • {}",
                            self.video_width,
                            self.video_height,
                            look_name,
                            self.gui_render_quality.name()
                        )
                    }
                };
                ui.weak(summary_line);
                ui.add_space(2.0);

                match self.render_kind {
                    StudioRenderKind::Animation => {
                        if Self::icon_button(
                            ui,
                            UiIcon::Film,
                            "Start Video Render",
                            true,
                            Some(egui::vec2(ui.available_width(), 35.0)),
                        )
                        .on_hover_text(
                            "Run full offline fluid simulation & studio rendering and play the finished video right here",
                        )
                        .clicked()
                        {
                            self.central_view = CentralView::RenderResult;
                            self.start_gui_mp4_render_with_ctx(Some(ctx.clone()));
                        }
                    }
                    StudioRenderKind::StillFrame => {
                        if Self::icon_button(
                            ui,
                            UiIcon::Camera,
                            "Render High-Res Photo (F12)",
                            true,
                            Some(egui::vec2(ui.available_width(), 35.0)),
                        )
                        .on_hover_text(
                            "Render a high-resolution studio photograph and display the finished image",
                        )
                        .clicked()
                        {
                            self.central_view = CentralView::RenderResult;
                            self.start_gui_still_render(
                                Some(ctx.clone()),
                                self.render_auto_save_png,
                            );
                        }
                    }
                }
            }

            if has_result && !is_running {
                ui.separator();
                egui::Grid::new("render_result_summary_grid")
                    .num_columns(2)
                    .spacing([10.0, 3.0])
                    .show(ui, |ui| {
                        ui.weak("Result:");
                        if anim_len > 0 || bg_kind == StudioRenderKind::Animation {
                            ui.monospace(format!(
                                "{}×{} • {} frames ({:.1}s)",
                                out_w,
                                out_h,
                                done.max(anim_len),
                                elapsed
                            ));
                        } else {
                            ui.monospace(format!(
                                "{}×{} photo ({:.2}s)",
                                out_w, out_h, elapsed
                            ));
                        }
                        ui.end_row();

                        if sim_pts > 0 {
                            ui.weak("Fluid:");
                            ui.monospace(format!(
                                "{} particles ({})",
                                Self::format_particle_count(sim_pts),
                                qual.slug()
                            ));
                            ui.end_row();
                        }

                        if !backend_lbl.is_empty() {
                            ui.weak("Engine:");
                            ui.colored_label(
                                egui::Color32::from_rgb(114, 198, 255),
                                &backend_lbl,
                            );
                            ui.end_row();
                        }
                    });

                ui.add_space(3.0);
                ui.horizontal(|ui| {
                    let half_w = (ui.available_width() - 6.0) * 0.5;
                    if Self::icon_button(
                        ui,
                        UiIcon::Save,
                        "Save Image As...",
                        false,
                        Some(egui::vec2(half_w, 26.0)),
                    )
                    .on_hover_text("Choose where to save the displayed frame as a lossless PNG image")
                    .clicked()
                    {
                        self.spawn_render_file_dialog(ctx.clone(), 2);
                    }

                    let folder_target = saved_path
                        .clone()
                        .unwrap_or_else(|| self.export_path.clone());
                    if Self::icon_button(
                        ui,
                        UiIcon::Folder,
                        "Open Folder",
                        false,
                        Some(egui::vec2(half_w, 26.0)),
                    )
                    .on_hover_text("Open the output folder in your computer's file manager")
                    .clicked()
                    {
                        Self::open_in_file_manager(&folder_target);
                    }
                });

                let (view_ic, view_lbl) = if self.central_view == CentralView::RenderResult {
                    (UiIcon::Cube3D, "Back to Live 3D Editor")
                } else {
                    (UiIcon::Image, "Show Rendered Result")
                };
                if Self::icon_button(
                    ui,
                    view_ic,
                    view_lbl,
                    self.central_view == CentralView::RenderResult,
                    Some(egui::vec2(ui.available_width(), 26.0)),
                )
                .clicked()
                {
                    self.central_view = match self.central_view {
                        CentralView::RenderResult => CentralView::LiveViewport,
                        CentralView::LiveViewport => CentralView::RenderResult,
                    };
                }

                if let Some(ref p) = saved_path {
                    ui.horizontal(|ui| {
                        ui.weak("Saved:");
                        ui.monospace(
                            egui::RichText::new(p)
                                .size(11.0)
                                .color(egui::Color32::from_rgb(120, 224, 158)),
                        );
                        if Self::icon_button(
                            ui,
                            UiIcon::Copy,
                            "",
                            false,
                            Some(egui::vec2(22.0, 20.0)),
                        )
                        .on_hover_text("Copy file path to clipboard")
                        .clicked()
                        {
                            ctx.copy_text(p.clone());
                            self.status_message = format!("Copied path '{}'", p);
                        }
                    });
                }
            }
        });

        ui.add_space(6.0);

        // 4. Collapsed Power-User & CLI Drawer (keeps UI clean for beginners!)
        egui::CollapsingHeader::new("Advanced & CLI Options")
            .default_open(false)
            .show(ui, |ui| {
                Self::section_card(ui, "Custom Size, 3D Mesh Export & CLI", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Exact Pixels:");
                        let mut w_i = self.video_width as u32;
                        let mut h_i = self.video_height as u32;
                        if ui
                            .add(
                                egui::DragValue::new(&mut w_i)
                                    .range(320..=7680)
                                    .speed(4)
                                    .suffix(" px"),
                            )
                            .changed()
                        {
                            self.video_width = ((w_i as usize + 1) & !1).clamp(320, 7680);
                            self.guided_render_preset = GuidedRenderPreset::Custom;
                        }
                        ui.label("×");
                        if ui
                            .add(
                                egui::DragValue::new(&mut h_i)
                                    .range(240..=4320)
                                    .speed(4)
                                    .suffix(" px"),
                            )
                            .changed()
                        {
                            self.video_height = ((h_i as usize + 1) & !1).clamp(240, 4320);
                            self.guided_render_preset = GuidedRenderPreset::Custom;
                        }
                    });

                    if self.render_kind == StudioRenderKind::Animation {
                        ui.horizontal(|ui| {
                            ui.label("Start Video:");
                            if ui
                                .selectable_label(
                                    self.render_anim_start == AnimationStartSource::PresetStart,
                                    "Scene Start (t=0)",
                                )
                                .clicked()
                            {
                                self.render_anim_start = AnimationStartSource::PresetStart;
                            }
                            if ui
                                .selectable_label(
                                    self.render_anim_start == AnimationStartSource::CurrentState,
                                    "Current Fluid",
                                )
                                .on_hover_text(
                                    "Continues from live particles when Fluid Detail is 'Same as 3D Editor'",
                                )
                                .clicked()
                            {
                                self.render_anim_start = AnimationStartSource::CurrentState;
                            }
                        });

                        if !self.is_2d {
                            ui.add(
                                egui::Slider::new(&mut self.camera_orbit_speed_deg, -60.0..=60.0)
                                    .text("Orbit Speed")
                                    .suffix(" °/s"),
                            );
                        }

                        ui.checkbox(
                            &mut self.render_also_export_obj,
                            "Also export 3D Surface Mesh sequence (.obj)",
                        )
                        .on_hover_text("Writes fluid_surface_00000.obj ... into the Export folder for Blender / Houdini");
                        ui.checkbox(
                            &mut self.render_also_export_ply,
                            "Also export Particle Cloud sequence (.ply)",
                        );
                    }

                    ui.checkbox(
                        &mut self.use_gpu,
                        "Hardware Vulkan GPU Acceleration",
                    )
                    .on_hover_text(
                        "Uses Vulkan GPU simulation, GPU Marching Cubes & offscreen SSAA rasterizer (uncheck for multi-core CPU)",
                    );

                    ui.separator();
                    ui.weak("Equivalent Terminal (CLI) Command:");
                    let mut cli_cmd = self.generate_render_tab_cli_command();
                    ui.add(
                        egui::TextEdit::multiline(&mut cli_cmd)
                            .font(egui::TextStyle::Monospace)
                            .desired_rows(3)
                            .desired_width(f32::INFINITY),
                    );
                    if Self::icon_button(ui, UiIcon::Copy, "Copy CLI Command", false, None)
                        .clicked()
                    {
                        ctx.copy_text(cli_cmd);
                        self.status_message =
                            "Copied equivalent CLI render command to clipboard!".to_string();
                    }
                });
            });
    }

    fn draw_render_cli_tab(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // 1. Single-Frame Studio Snapshot & Geometry Export (PNG / PLY / OBJ)
        Self::section_card(ui, "Frame Snapshot & 3D Geometry Export", |ui| {
            ui.horizontal(|ui| {
                ui.label("Export Dir:");
                ui.text_edit_singleline(&mut self.export_path);
            });

            if Self::icon_button(
                ui,
                UiIcon::Image,
                &format!(
                    "Capture Studio PNG ({}x{})",
                    self.video_width, self.video_height
                ),
                false,
                Some(egui::vec2(ui.available_width(), 27.0)),
            )
            .on_hover_text("Render current frame with full studio SSAA, shadows & optics to a lossless PNG file (Shortcut: P)")
            .clicked()
            {
                self.export_current_png();
            }

            ui.horizontal(|ui| {
                let w = (ui.available_width() - 6.0) * 0.5;
                if Self::icon_button(
                    ui,
                    UiIcon::Save,
                    "Save Cloud (.ply)",
                    false,
                    Some(egui::vec2(w, 25.0)),
                )
                .clicked()
                {
                    self.export_current_frame();
                }
                if Self::icon_button(
                    ui,
                    UiIcon::Mesh,
                    "Save Mesh (.obj)",
                    false,
                    Some(egui::vec2(w, 25.0)),
                )
                .clicked()
                {
                    self.export_current_surface_obj();
                }
            });

            ui.checkbox(&mut self.export_sequence, "Auto-record PLY sequence while playing");
            if self.export_sequence {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 185, 65),
                    format!("Recording frame {:05}", self.exported_frames_count),
                );
            }
        });

        ui.add_space(6.0);

        // 2. Quick Link to Dedicated Studio Render Tab
        Self::section_card(ui, "Studio Image & Video Rendering", |ui| {
            ui.weak(
                "Use the Render tab to configure resolution & optics, preview rendered frames, and play back animations right inside the viewport.",
            );
            ui.add_space(3.0);
            if Self::icon_button(
                ui,
                UiIcon::Camera,
                "Open Studio Render Tab",
                false,
                Some(egui::vec2(ui.available_width(), 28.0)),
            )
            .clicked()
            {
                self.active_tab = InspectorTab::Render;
                self.central_view = CentralView::RenderResult;
            }
        });

        ui.add_space(6.0);

        // 3. CLI Command Builder
        Self::section_card(ui, "Runnable CLI Command Builder", |ui| {
            ui.horizontal_wrapped(|ui| {
                for &m in CliCommandMode::ALL {
                    if ui.selectable_label(self.cli_mode == m, m.name()).clicked() {
                        self.cli_mode = m;
                    }
                }
            });

            let mut cmd = self.generate_cli_command();
            ui.add(
                egui::TextEdit::multiline(&mut cmd)
                    .font(egui::TextStyle::Monospace)
                    .desired_rows(4)
                    .desired_width(f32::INFINITY),
            );

            ui.horizontal(|ui| {
                if Self::icon_button(ui, UiIcon::Copy, "Copy CLI Command", false, None).clicked() {
                    ctx.copy_text(cmd.clone());
                    self.status_message = "Copied CLI command to clipboard!".to_string();
                }
                if Self::icon_button(ui, UiIcon::Mesh, "Copy Mesh MP4 CLI", false, None).clicked()
                {
                    let prev_mode = self.cli_mode;
                    let prev_rm = self.render_mode;
                    self.cli_mode = CliCommandMode::Mp4Video;
                    self.render_mode = RenderMode::SurfaceMesh;
                    let mesh_cmd = self.generate_cli_command();
                    self.cli_mode = prev_mode;
                    self.render_mode = prev_rm;
                    ctx.copy_text(mesh_cmd);
                    self.status_message =
                        "Copied Surface-Mesh MP4 CLI render command!".to_string();
                }
            });
        });
    }
}

impl eframe::App for LiquidApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            Self::apply_studio_theme(ctx);
            self.theme_applied = true;
        }

        // Poll native OS file dialogs for Render output paths & "Save Image As..."
        if let Ok(mut g) = self.render_mp4_browse_result.lock() {
            if let Some(mut chosen) = g.take() {
                if std::path::Path::new(&chosen).extension().is_none() {
                    chosen.push_str(".mp4");
                }
                self.cli_mp4_output = chosen;
                self.render_save_mp4 = true;
            }
        }
        if let Ok(mut g) = self.render_png_browse_result.lock() {
            if let Some(mut chosen) = g.take() {
                if std::path::Path::new(&chosen).extension().is_none() {
                    chosen.push_str(".png");
                }
                self.png_output_path = chosen;
                self.render_auto_save_png = true;
            }
        }
        let save_as_png_now = {
            if let Ok(mut g) = self.render_save_result_png_slot.lock() {
                g.take()
            } else {
                None
            }
        };
        if let Some(mut chosen) = save_as_png_now {
            if std::path::Path::new(&chosen).extension().is_none() {
                chosen.push_str(".png");
            }
            self.save_rendered_result_png_to_path(std::path::Path::new(&chosen));
        }

        // Sync background studio render progress, messages, and animation auto-play state
        let (
            bg_is_running,
            bg_kind,
            bg_done,
            bg_total,
            bg_elapsed,
            bg_phase,
            bg_sim_pts,
            bg_has_still,
            bg_anim_len,
            bg_fps,
            bg_out_w,
            bg_out_h,
            bg_qual,
            bg_backend,
            bg_saved_path,
        ) = {
            let mut finished_msg = None;
            let mut start_playback = false;
            let mut snap = (
                false,
                StudioRenderKind::StillFrame,
                0usize,
                0usize,
                0.0f32,
                String::new(),
                0usize,
                false,
                0usize,
                60u32,
                self.video_width,
                self.video_height,
                self.gui_render_quality,
                String::new(),
                None,
            );
            if let Ok(mut g) = self.bg_render.lock() {
                finished_msg = g.finished_message.take();
                if g.just_finished_anim {
                    g.just_finished_anim = false;
                    start_playback = true;
                }
                snap = (
                    g.is_running,
                    g.kind,
                    g.completed_frames,
                    g.total_frames,
                    g.elapsed_secs,
                    g.phase_label.clone(),
                    g.sim_particles_count,
                    g.full_res_frame.is_some(),
                    g.anim_frames.len(),
                    g.fps.max(1),
                    g.output_width,
                    g.output_height,
                    g.quality,
                    g.backend_label.clone(),
                    g.saved_path.clone(),
                );
            }
            if let Some(msg) = finished_msg {
                self.status_message = msg;
            }
            if start_playback && snap.8 > 1 {
                self.central_view = CentralView::RenderResult;
                self.anim_play_idx = 0;
                self.anim_accum_secs = 0.0;
                self.anim_playing = true;
                self.last_anim_tick = Instant::now();
            }
            snap
        };

        if bg_is_running {
            if bg_anim_len > 0 {
                self.anim_play_idx = bg_anim_len - 1;
            }
            ctx.request_repaint();
        }

        // Advance rendered animation playback clock
        let now_anim = Instant::now();
        let anim_wall_dt = now_anim
            .duration_since(self.last_anim_tick)
            .as_secs_f32()
            .min(0.1);
        self.last_anim_tick = now_anim;
        if self.central_view == CentralView::RenderResult
            && self.anim_playing
            && !bg_is_running
            && bg_anim_len > 1
        {
            self.anim_accum_secs += anim_wall_dt * self.anim_speed.max(0.05);
            let frame_dur = 1.0 / bg_fps as f32;
            while self.anim_accum_secs >= frame_dur {
                self.anim_accum_secs -= frame_dur;
                if self.anim_play_idx + 1 < bg_anim_len {
                    self.anim_play_idx += 1;
                } else if self.anim_loop {
                    self.anim_play_idx = 0;
                } else {
                    self.anim_playing = false;
                    self.anim_accum_secs = 0.0;
                    break;
                }
            }
            ctx.request_repaint();
        }

        // Studio keyboard shortcuts (only when not typing into a text input widget)
        if !ctx.wants_keyboard_input() {
            let mut key_space = false;
            let mut key_step = false;
            let mut key_reset = false;
            let mut key_cam = false;
            let mut key_mesh = false;
            let mut key_grid = false;
            let mut key_wire = false;
            let mut key_orbit = false;
            let mut key_png = false;
            let mut key_f12 = false;
            let mut key_help = false;
            let mut key_left = false;
            let mut key_right = false;

            ctx.input(|i| {
                if i.key_pressed(egui::Key::Space) {
                    key_space = true;
                }
                if i.key_pressed(egui::Key::Period) {
                    key_step = true;
                }
                if i.key_pressed(egui::Key::R) && !i.modifiers.command && !i.modifiers.ctrl {
                    key_reset = true;
                }
                if i.key_pressed(egui::Key::C) && !i.modifiers.command && !i.modifiers.ctrl {
                    key_cam = true;
                }
                if i.key_pressed(egui::Key::M) {
                    key_mesh = true;
                }
                if i.key_pressed(egui::Key::G) {
                    key_grid = true;
                }
                if i.key_pressed(egui::Key::B) {
                    key_wire = true;
                }
                if i.key_pressed(egui::Key::O) {
                    key_orbit = true;
                }
                if i.key_pressed(egui::Key::P) {
                    key_png = true;
                }
                if i.key_pressed(egui::Key::F12) {
                    key_f12 = true;
                }
                if i.key_pressed(egui::Key::ArrowLeft) {
                    key_left = true;
                }
                if i.key_pressed(egui::Key::ArrowRight) {
                    key_right = true;
                }
                if i.key_pressed(egui::Key::F1)
                    || (i.key_pressed(egui::Key::Slash) && i.modifiers.shift)
                {
                    key_help = true;
                }
            });

            if key_space {
                if self.central_view == CentralView::RenderResult
                    && !bg_is_running
                    && bg_anim_len > 1
                {
                    if !self.anim_playing && self.anim_play_idx + 1 >= bg_anim_len {
                        self.anim_play_idx = 0;
                    }
                    self.anim_playing = !self.anim_playing;
                    self.last_anim_tick = Instant::now();
                } else {
                    self.is_playing = !self.is_playing;
                }
            }
            if key_left
                && self.central_view == CentralView::RenderResult
                && !bg_is_running
                && bg_anim_len > 1
            {
                self.anim_playing = false;
                self.anim_play_idx = if self.anim_play_idx > 0 {
                    self.anim_play_idx - 1
                } else {
                    bg_anim_len - 1
                };
            }
            if key_right
                && self.central_view == CentralView::RenderResult
                && !bg_is_running
                && bg_anim_len > 1
            {
                self.anim_playing = false;
                self.anim_play_idx = (self.anim_play_idx + 1) % bg_anim_len;
            }
            if key_step {
                let dt = self.fixed_dt * self.sim_speed;
                self.step_simulation_once(dt);
                ctx.request_repaint();
            }
            if key_reset {
                self.load_preset(self.current_preset);
            }
            if key_cam {
                if self.central_view == CentralView::RenderResult {
                    self.render_zoom = 0.0;
                    self.render_pan = egui::Vec2::ZERO;
                } else {
                    self.reset_camera_framing();
                }
            }
            if key_mesh {
                self.render_mode = match self.render_mode {
                    RenderMode::Particles => {
                        self.mesh_dirty = true;
                        RenderMode::SurfaceMesh
                    }
                    RenderMode::SurfaceMesh => RenderMode::Particles,
                };
            }
            if key_grid {
                self.mesh_config.show_grid = !self.mesh_config.show_grid;
            }
            if key_wire {
                self.mesh_config.show_wireframe = !self.mesh_config.show_wireframe;
            }
            if key_orbit && !self.is_2d {
                self.camera_orbit_speed_deg = if self.camera_orbit_speed_deg.abs() > 0.5 {
                    0.0
                } else {
                    18.0
                };
            }
            if key_png {
                self.export_current_png();
            }
            if key_f12 {
                self.active_tab = InspectorTab::Render;
                self.central_view = CentralView::RenderResult;
                self.start_gui_still_render(Some(ctx.clone()), self.render_auto_save_png);
            }
            if key_help {
                self.show_shortcuts_modal = !self.show_shortcuts_modal;
            }
        }

        // Turntable camera auto-orbit
        let cam_dt = self.last_camera_tick.elapsed().as_secs_f32().min(0.1);
        self.last_camera_tick = Instant::now();
        if !self.is_2d && self.camera_orbit_speed_deg.abs() > 0.05 {
            self.camera.azimuth += self.camera_orbit_speed_deg.to_radians() * cam_dt;
            ctx.request_repaint();
        }

        // Calculate FPS smoothly
        self.frame_counter += 1;
        let elapsed = self.last_frame_time.elapsed().as_secs_f32();
        if elapsed >= 0.4 {
            let instant_fps = self.frame_counter as f32 / elapsed;
            self.current_fps = self.current_fps * 0.35 + instant_fps * 0.65;
            self.frame_counter = 0;
            self.last_frame_time = Instant::now();
        }

        // Simulation update step
        if self.is_playing {
            let effective_dt = self.fixed_dt * self.sim_speed;
            self.step_simulation_once(effective_dt);

            if self.export_sequence {
                self.export_current_frame();
            }

            ctx.request_repaint();
        }

        // 1. Studio Top Transport & Telemetry Bar
        egui::TopBottomPanel::top("studio_top_bar")
            .exact_height(42.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Brand logo icon + title
                    let (logo_rect, _) =
                        ui.allocate_exact_size(egui::vec2(18.0, 20.0), egui::Sense::hover());
                    Self::paint_icon(
                        ui.painter(),
                        logo_rect.center(),
                        UiIcon::Wave,
                        egui::Color32::from_rgb(86, 188, 255),
                    );
                    ui.label(
                        egui::RichText::new("LiquidRust")
                            .strong()
                            .size(15.5)
                            .color(egui::Color32::from_rgb(236, 245, 255)),
                    );

                    // Version pill / Update badge right next to app title
                    let upd_snap = self.updater.snapshot();
                    match &upd_snap.phase {
                        UpdatePhase::UpdateAvailable(info) => {
                            let upd_btn = egui::Button::new(
                                egui::RichText::new(format!("Update {}", info.tag_name))
                                    .strong()
                                    .size(11.5)
                                    .color(egui::Color32::from_rgb(10, 24, 18)),
                            )
                            .fill(egui::Color32::from_rgb(74, 222, 128));
                            if ui
                                .add(upd_btn)
                                .on_hover_text(format!(
                                    "New release {} is available on GitHub! Click to update & restart.",
                                    info.tag_name
                                ))
                                .clicked()
                            {
                                self.updater.open_modal();
                            }
                        }
                        UpdatePhase::Downloading {
                            downloaded_bytes,
                            total_bytes,
                            ..
                        } => {
                            let pct = if *total_bytes > 0 {
                                (*downloaded_bytes * 100) / *total_bytes
                            } else {
                                0
                            };
                            let dl_btn = egui::Button::new(
                                egui::RichText::new(format!("DL {}%", pct))
                                    .strong()
                                    .size(11.5)
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(egui::Color32::from_rgb(22, 108, 196));
                            if ui.add(dl_btn).clicked() {
                                self.updater.open_modal();
                            }
                        }
                        _ => {
                            let ver_btn = egui::Button::new(
                                egui::RichText::new(format!("v{}", CURRENT_VERSION))
                                    .monospace()
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(148, 178, 214)),
                            )
                            .fill(egui::Color32::from_rgb(20, 28, 42));
                            if ui
                                .add(ver_btn)
                                .on_hover_text("Check GitHub Releases for updates")
                                .clicked()
                            {
                                self.updater.spawn_check(Some(ctx.clone()), true);
                            }
                        }
                    }

                    if Self::icon_button(ui, UiIcon::Cube3D, "3D", !self.is_2d, None)
                        .on_hover_text("Switch to 3D Volumetric simulation")
                        .clicked()
                        && self.is_2d
                    {
                        self.set_2d(false);
                    }
                    if Self::icon_button(ui, UiIcon::Plane2D, "2D", self.is_2d, None)
                        .on_hover_text("Switch to 2D Planar simulation")
                        .clicked()
                        && !self.is_2d
                    {
                        self.set_2d(true);
                    }

                    ui.separator();

                    let (play_icon, play_lbl) = if self.is_playing {
                        (UiIcon::Pause, "Pause")
                    } else {
                        (UiIcon::Play, "Play")
                    };
                    if Self::icon_button(ui, play_icon, play_lbl, self.is_playing, None)
                        .on_hover_text("Play / Pause simulation (Space)")
                        .clicked()
                    {
                        self.is_playing = !self.is_playing;
                    }

                    if Self::icon_button(ui, UiIcon::Step, "Step", false, None)
                        .on_hover_text("Advance single simulation frame (.)")
                        .clicked()
                    {
                        let dt = self.fixed_dt * self.sim_speed;
                        self.step_simulation_once(dt);
                        ctx.request_repaint();
                    }

                    if Self::icon_button(ui, UiIcon::Reset, "Reset", false, None)
                        .on_hover_text("Reset scene particles (R)")
                        .clicked()
                    {
                        self.load_preset(self.current_preset);
                    }

                    if Self::icon_button(ui, UiIcon::Camera, "Cam", false, None)
                        .on_hover_text("Reset camera framing (C)")
                        .clicked()
                    {
                        self.reset_camera_framing();
                    }

                    ui.separator();

                    egui::ComboBox::from_id_salt("top_preset_combo")
                        .selected_text(self.current_preset.name())
                        .width(155.0)
                        .show_ui(ui, |ui| {
                            for &preset in ScenePreset::ALL {
                                if ui
                                    .selectable_value(
                                        &mut self.current_preset,
                                        preset,
                                        preset.name(),
                                    )
                                    .clicked()
                                {
                                    self.load_preset(preset);
                                }
                            }
                        });

                    ui.separator();

                    // Quick Render Mode Toggle right in top bar
                    if Self::icon_button(
                        ui,
                        UiIcon::Spheres,
                        "Spheres",
                        self.render_mode == RenderMode::Particles,
                        None,
                    )
                    .on_hover_text("Instanced SPH sphere shading (M)")
                    .clicked()
                    {
                        self.render_mode = RenderMode::Particles;
                    }
                    if Self::icon_button(
                        ui,
                        UiIcon::Mesh,
                        "Mesh",
                        self.render_mode == RenderMode::SurfaceMesh,
                        None,
                    )
                    .on_hover_text("3D Marching Cubes liquid surface mesh (M)")
                    .clicked()
                    {
                        self.render_mode = RenderMode::SurfaceMesh;
                        self.mesh_dirty = true;
                    }

                    ui.separator();

                    for &(spd, lbl) in &[(0.25f32, "0.25x"), (0.5f32, "0.5x"), (1.0f32, "1.0x")] {
                        if ui
                            .selectable_label((self.sim_speed - spd).abs() < 0.03, lbl)
                            .clicked()
                        {
                            self.sim_speed = spd;
                        }
                    }

                    // Right-aligned zero-jitter monospace telemetry pills + quick actions
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if Self::icon_button(
                            ui,
                            UiIcon::Help,
                            "",
                            self.show_shortcuts_modal,
                            Some(egui::vec2(24.0, 24.0)),
                        )
                        .on_hover_text("Keyboard & Mouse Shortcuts (? / F1)")
                        .clicked()
                        {
                            self.show_shortcuts_modal = !self.show_shortcuts_modal;
                        }

                        if Self::icon_button(ui, UiIcon::Image, "PNG", false, None)
                            .on_hover_text("Save Studio PNG screenshot to exports folder (P / F12)")
                            .clicked()
                        {
                            self.export_current_png();
                        }

                        if Self::icon_button(ui, UiIcon::Copy, "CLI", false, None)
                            .on_hover_text("Copy runnable CLI command to clipboard")
                            .clicked()
                        {
                            ctx.copy_text(self.generate_cli_command());
                            self.status_message = "Copied CLI command to clipboard!".to_string();
                        }

                        Self::draw_telemetry_pill(
                            ui,
                            format!("{:>5.1} ms", self.last_step_time_ms),
                            egui::Color32::from_rgb(130, 198, 255),
                        );

                        if self.render_mode == RenderMode::SurfaceMesh {
                            Self::draw_telemetry_pill(
                                ui,
                                format!("{:>7} tris", self.cached_surface_mesh.triangle_count()),
                                egui::Color32::from_rgb(168, 226, 255),
                            );
                        } else {
                            Self::draw_telemetry_pill(
                                ui,
                                format!("{:>7} pts", self.solver.particle_count()),
                                egui::Color32::from_rgb(190, 210, 236),
                            );
                        }

                        let fps_col = if self.current_fps >= 50.0 {
                            egui::Color32::from_rgb(82, 224, 150)
                        } else if self.current_fps >= 25.0 {
                            egui::Color32::from_rgb(255, 198, 86)
                        } else {
                            egui::Color32::from_rgb(255, 110, 110)
                        };
                        Self::draw_telemetry_pill(
                            ui,
                            format!("{:>3.0} FPS", self.current_fps.clamp(0.0, 999.0)),
                            fps_col,
                        );
                    });
                });
            });

        // 2. Studio Bottom Status Bar
        egui::TopBottomPanel::bottom("studio_status_bar")
            .exact_height(24.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    let (b_icon, backend_str) = if self.use_gpu && self.gpu_solver.is_some() {
                        (UiIcon::Bolt, format!("Vulkan: {}", self.gpu_device_name))
                    } else {
                        (UiIcon::Sliders, "CPU (Rayon Multi-Core)".to_string())
                    };
                    let (ic_r, _) =
                        ui.allocate_exact_size(egui::vec2(13.0, 14.0), egui::Sense::hover());
                    Self::paint_icon(
                        ui.painter(),
                        ic_r.center(),
                        b_icon,
                        egui::Color32::from_rgb(112, 184, 255),
                    );
                    ui.label(
                        egui::RichText::new(backend_str)
                            .size(11.0)
                            .color(egui::Color32::from_rgb(112, 184, 255)),
                    );
                    ui.separator();
                    ui.label(
                        egui::RichText::new(&self.status_message)
                            .size(11.0)
                            .color(egui::Color32::from_rgb(194, 206, 226)),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(if self.is_2d {
                            "Space: Play • R: Reset • M: Mesh • F12: Render • P: PNG • ?: Help • L-Drag: Pull"
                        } else {
                            "Space: Play • R: Reset • M: Mesh • O: Orbit • F12: Render • P: PNG • ?: Help • R-Drag: Orbit"
                        });
                    });
                });
            });

        // 3. Left Studio Inspector Panel with 5 Organized Tabs (Scene | Physics | Visuals | Render | Export)
        egui::SidePanel::left("studio_inspector")
            .default_width(372.0)
            .min_width(352.0)
            .max_width(450.0)
            .show(ctx, |ui| {
                ui.add_space(3.0);
                ui.horizontal(|ui| {
                    let sp = ui.spacing().item_spacing.x;
                    let tab_w = ((ui.available_width() - sp * 4.0) / 5.0).max(58.0);
                    for &(tab, icon, label) in &[
                        (InspectorTab::Scene, UiIcon::Wave, "Scene"),
                        (InspectorTab::Physics, UiIcon::Sliders, "Physics"),
                        (InspectorTab::Surface, UiIcon::Mesh, "Visuals"),
                        (InspectorTab::Render, UiIcon::Camera, "Render"),
                        (InspectorTab::RenderCli, UiIcon::Save, "Export"),
                    ] {
                        if Self::inspector_tab_button(
                            ui,
                            icon,
                            label,
                            self.active_tab == tab,
                            tab_w,
                        )
                        .clicked()
                        {
                            self.active_tab = tab;
                            match tab {
                                InspectorTab::Render => {
                                    if bg_is_running || bg_has_still || bg_anim_len > 0 {
                                        self.central_view = CentralView::RenderResult;
                                    }
                                }
                                InspectorTab::Scene
                                | InspectorTab::Physics
                                | InspectorTab::Surface => {
                                    if !bg_is_running {
                                        self.central_view = CentralView::LiveViewport;
                                    }
                                }
                                InspectorTab::RenderCli => {}
                            }
                        }
                    }
                });
                ui.separator();

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        match self.active_tab {
                            InspectorTab::Scene => self.draw_scene_tab(ui),
                            InspectorTab::Physics => self.draw_physics_tab(ui),
                            InspectorTab::Surface => self.draw_surface_tab(ui),
                            InspectorTab::Render => self.draw_render_tab(ctx, ui),
                            InspectorTab::RenderCli => self.draw_render_cli_tab(ctx, ui),
                        }
                        ui.add_space(10.0);
                    });
            });

        // 3b. Central Workspace View Switcher Bar (Live 3D Viewport <-> Render Result)
        egui::TopBottomPanel::top("central_workspace_bar")
            .exact_height(32.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    let live_icon = if self.is_2d {
                        UiIcon::Plane2D
                    } else {
                        UiIcon::Cube3D
                    };
                    if Self::icon_button(
                        ui,
                        live_icon,
                        "Live 3D Editor",
                        self.central_view == CentralView::LiveViewport,
                        None,
                    )
                    .on_hover_text("Interactive real-time 3D fluid editor")
                    .clicked()
                    {
                        self.central_view = CentralView::LiveViewport;
                    }

                    let res_lbl = if bg_anim_len > 1 {
                        format!("Render Result ({} frames)", bg_anim_len)
                    } else {
                        "Render Result".to_string()
                    };
                    if Self::icon_button(
                        ui,
                        UiIcon::Image,
                        &res_lbl,
                        self.central_view == CentralView::RenderResult,
                        None,
                    )
                    .on_hover_text("Inspect rendered photo or watch rendered video animation (F12)")
                    .clicked()
                    {
                        self.central_view = CentralView::RenderResult;
                    }

                    if bg_is_running {
                        ui.separator();
                        ui.spinner();
                        let eta_suffix = Self::format_eta(bg_done, bg_total, bg_elapsed)
                            .map(|s| format!(" • {}", s))
                            .unwrap_or_default();
                        let prog_str = if bg_total > 1 {
                            format!("Rendering {}/{}{}", bg_done, bg_total, eta_suffix)
                        } else if !bg_phase.is_empty() {
                            bg_phase.clone()
                        } else {
                            "Rendering...".to_string()
                        };
                        ui.colored_label(egui::Color32::from_rgb(108, 204, 255), prog_str);
                    }

                    if self.central_view == CentralView::RenderResult {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if let Some(ref p) = bg_saved_path {
                                if Self::icon_button(ui, UiIcon::Folder, "Folder", false, None)
                                    .on_hover_text("Open saved file folder in your file manager")
                                    .clicked()
                                {
                                    Self::open_in_file_manager(p);
                                }
                            }

                            if (bg_has_still || bg_anim_len > 0)
                                && Self::icon_button(ui, UiIcon::Save, "Save As...", false, None)
                                    .on_hover_text("Save displayed image anywhere on your computer (.png)")
                                    .clicked()
                            {
                                self.spawn_render_file_dialog(ctx.clone(), 2);
                            }

                            if !bg_is_running {
                                let (rr_ic, rr_lbl) = match self.render_kind {
                                    StudioRenderKind::StillFrame => {
                                        (UiIcon::Camera, "Render Photo")
                                    }
                                    StudioRenderKind::Animation => (UiIcon::Film, "Render Video"),
                                };
                                if Self::icon_button(ui, rr_ic, rr_lbl, false, None)
                                    .on_hover_text("Start render using your Render tab settings")
                                    .clicked()
                                {
                                    match self.render_kind {
                                        StudioRenderKind::StillFrame => {
                                            self.start_gui_still_render(
                                                Some(ctx.clone()),
                                                self.render_auto_save_png,
                                            );
                                        }
                                        StudioRenderKind::Animation => {
                                            self.start_gui_mp4_render_with_ctx(Some(ctx.clone()));
                                        }
                                    }
                                }
                            }

                            if ui
                                .selectable_label(
                                    (self.render_zoom - 1.0).abs() < 0.02,
                                    "1:1",
                                )
                                .on_hover_text("Actual 100% pixel resolution")
                                .clicked()
                            {
                                self.render_zoom = 1.0;
                                self.render_pan = egui::Vec2::ZERO;
                            }
                            if ui
                                .selectable_label(self.render_zoom <= 0.0, "Fit")
                                .on_hover_text("Fit rendered frame inside workspace (C / Double-Click)")
                                .clicked()
                            {
                                self.render_zoom = 0.0;
                                self.render_pan = egui::Vec2::ZERO;
                            }

                            if bg_has_still || bg_anim_len > 0 {
                                let pts_suffix = if bg_sim_pts > 0 {
                                    format!(" • {} pts", Self::format_particle_count(bg_sim_pts))
                                } else {
                                    String::new()
                                };
                                Self::draw_telemetry_pill(
                                    ui,
                                    format!(
                                        "{}×{} • {}{}",
                                        bg_out_w,
                                        bg_out_h,
                                        bg_qual.slug(),
                                        pts_suffix
                                    ),
                                    egui::Color32::from_rgb(154, 214, 255),
                                );
                                if !bg_backend.is_empty() {
                                    Self::draw_telemetry_pill(
                                        ui,
                                        bg_backend.clone(),
                                        egui::Color32::from_rgb(112, 224, 168),
                                    );
                                }
                            }
                        });
                    }
                });
            });

        // 3c. Rendered Animation Timeline & Transport Bar (shown when Render Result has an animation)
        if self.central_view == CentralView::RenderResult && bg_anim_len > 1 {
            egui::TopBottomPanel::bottom("render_playback_bar")
                .exact_height(36.0)
                .show(ctx, |ui| {
                    ui.horizontal_centered(|ui| {
                        let (p_ic, p_lbl) = if self.anim_playing {
                            (UiIcon::Pause, "Pause")
                        } else {
                            (UiIcon::Play, "Play")
                        };
                        if Self::icon_button(
                            ui,
                            p_ic,
                            p_lbl,
                            self.anim_playing,
                            Some(egui::vec2(66.0, 24.0)),
                        )
                        .on_hover_text("Play / Pause rendered animation (Space)")
                        .clicked()
                            && !bg_is_running
                        {
                            if !self.anim_playing && self.anim_play_idx + 1 >= bg_anim_len {
                                self.anim_play_idx = 0;
                            }
                            self.anim_playing = !self.anim_playing;
                            self.last_anim_tick = Instant::now();
                        }

                        if Self::icon_button(
                            ui,
                            UiIcon::StepBack,
                            "",
                            false,
                            Some(egui::vec2(26.0, 24.0)),
                        )
                        .on_hover_text("Previous rendered frame (Left Arrow)")
                        .clicked()
                            && !bg_is_running
                        {
                            self.anim_playing = false;
                            self.anim_play_idx = if self.anim_play_idx > 0 {
                                self.anim_play_idx - 1
                            } else {
                                bg_anim_len - 1
                            };
                        }

                        if Self::icon_button(
                            ui,
                            UiIcon::Step,
                            "",
                            false,
                            Some(egui::vec2(26.0, 24.0)),
                        )
                        .on_hover_text("Next rendered frame (Right Arrow)")
                        .clicked()
                            && !bg_is_running
                        {
                            self.anim_playing = false;
                            self.anim_play_idx = (self.anim_play_idx + 1) % bg_anim_len;
                        }

                        if ui
                            .selectable_label(self.anim_loop, "Loop")
                            .on_hover_text("Loop animation playback continuously")
                            .clicked()
                        {
                            self.anim_loop = !self.anim_loop;
                        }

                        for &(spd, lbl) in &[(0.25f32, "0.25x"), (0.5f32, "0.5x"), (1.0f32, "1.0x")]
                        {
                            if ui
                                .selectable_label((self.anim_speed - spd).abs() < 0.03, lbl)
                                .clicked()
                            {
                                self.anim_speed = spd;
                            }
                        }

                        ui.separator();

                        let clamped_idx = self.anim_play_idx.min(bg_anim_len - 1);
                        let cur_secs = clamped_idx as f32 / bg_fps as f32;
                        let tot_secs = bg_anim_len as f32 / bg_fps as f32;
                        Self::draw_telemetry_pill(
                            ui,
                            format!(
                                "{:04}/{:04} • {:.2}s/{:.2}s",
                                clamped_idx + 1,
                                bg_anim_len,
                                cur_secs,
                                tot_secs
                            ),
                            egui::Color32::from_rgb(168, 214, 255),
                        );

                        let mut frame_one_based = clamped_idx + 1;
                        let slider_w = (ui.available_width() - 8.0).max(80.0);
                        ui.spacing_mut().slider_width = slider_w;
                        let resp = ui.add(
                            egui::Slider::new(&mut frame_one_based, 1..=bg_anim_len)
                                .show_value(false),
                        );
                        if resp.changed() && !bg_is_running {
                            self.anim_playing = false;
                            self.anim_play_idx = frame_one_based.saturating_sub(1);
                        }
                    });
                });
        }

        // 4. Central Workspace: EITHER Live OpenGL 3D Viewport OR Studio Render Result Viewer
        let bg_rgb = if self.central_view == CentralView::RenderResult {
            [8u8, 11, 18]
        } else {
            self.mesh_config.env_preset.viewport_bg_rgb8()
        };
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE.fill(egui::Color32::from_rgb(bg_rgb[0], bg_rgb[1], bg_rgb[2])),
            )
            .show(ctx, |ui| {
                if self.central_view == CentralView::RenderResult {
                    let available_size = ui.available_size();
                    let (rect, response) =
                        ui.allocate_exact_size(available_size, egui::Sense::click_and_drag());

                    if response.double_clicked() {
                        self.render_zoom = 0.0;
                        self.render_pan = egui::Vec2::ZERO;
                    }

                    // Fetch active rendered frame & upload to egui GPU texture only when changed
                    let frame_and_key = {
                        if let Ok(g) = self.bg_render.lock() {
                            if !g.anim_frames.is_empty() {
                                let idx = if g.is_running {
                                    g.anim_frames.len() - 1
                                } else {
                                    self.anim_play_idx.min(g.anim_frames.len() - 1)
                                };
                                if !g.is_running
                                    && idx + 1 == g.anim_frames.len()
                                    && g.full_res_frame.is_some()
                                {
                                    g.full_res_frame
                                        .clone()
                                        .map(|f| (f, (g.frame_revision, None)))
                                } else {
                                    Some((
                                        g.anim_frames[idx].clone(),
                                        (g.anim_generation, Some(idx)),
                                    ))
                                }
                            } else {
                                g.full_res_frame
                                    .clone()
                                    .map(|f| (f, (g.frame_revision, None)))
                            }
                        } else {
                            None
                        }
                    };

                    if let Some((frame, desired_key)) = frame_and_key {
                        if self.render_texture.is_none() || self.render_uploaded_key != desired_key
                        {
                            let color_img = egui::ColorImage::from_rgb(
                                [frame.width, frame.height],
                                &frame.rgb,
                            );
                            if let Some(ref mut tex) = self.render_texture {
                                tex.set(color_img, egui::TextureOptions::LINEAR);
                            } else {
                                self.render_texture = Some(ctx.load_texture(
                                    "studio_render_result",
                                    color_img,
                                    egui::TextureOptions::LINEAR,
                                ));
                            }
                            self.render_uploaded_key = desired_key;
                        }

                        if let Some(ref tex) = self.render_texture {
                            let ppp = ctx.pixels_per_point().max(0.5);
                            let img_pts_w = (bg_out_w.max(frame.width) as f32) / ppp;
                            let img_pts_h = (bg_out_h.max(frame.height) as f32) / ppp;
                            let pad = 24.0;
                            let fit_scale = ((rect.width() - pad).max(32.0) / img_pts_w.max(1.0))
                                .min((rect.height() - pad).max(32.0) / img_pts_h.max(1.0));

                            let scroll_y = ctx.input(|i| i.smooth_scroll_delta.y);
                            if scroll_y != 0.0 && response.hovered() {
                                let base_z = if self.render_zoom <= 0.0 {
                                    fit_scale
                                } else {
                                    self.render_zoom
                                };
                                self.render_zoom =
                                    (base_z * (1.0 + scroll_y * 0.0018)).clamp(0.08, 8.0);
                            }

                            if response.dragged_by(egui::PointerButton::Primary)
                                || response.dragged_by(egui::PointerButton::Middle)
                            {
                                self.render_pan += response.drag_delta();
                            }

                            let scale = if self.render_zoom <= 0.0 {
                                fit_scale
                            } else {
                                self.render_zoom
                            };
                            let draw_size = egui::vec2(img_pts_w * scale, img_pts_h * scale);
                            let img_rect =
                                egui::Rect::from_center_size(rect.center() + self.render_pan, draw_size);

                            let clipped = ui.painter().with_clip_rect(rect);
                            let shadow_rect = img_rect.expand(3.0).translate(egui::vec2(0.0, 3.0));
                            clipped.rect_filled(
                                shadow_rect,
                                egui::CornerRadius::same(4),
                                egui::Color32::from_black_alpha(140),
                            );
                            clipped.image(
                                tex.id(),
                                img_rect,
                                egui::Rect::from_min_max(
                                    egui::pos2(0.0, 0.0),
                                    egui::pos2(1.0, 1.0),
                                ),
                                egui::Color32::WHITE,
                            );
                            clipped.rect_stroke(
                                img_rect,
                                egui::CornerRadius::same(2),
                                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(44, 62, 92)),
                                egui::StrokeKind::Outside,
                            );
                        }
                    } else if !bg_is_running {
                        ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                            ui.centered_and_justified(|ui| {
                                ui.vertical_centered(|ui| {
                                    ui.add_space(rect.height() * 0.34);
                                    ui.label(
                                        egui::RichText::new("Ready to Render Your Fluid Scene")
                                            .size(17.0)
                                            .strong()
                                            .color(egui::Color32::from_rgb(208, 224, 248)),
                                    );
                                    ui.add_space(4.0);
                                    ui.label(
                                        egui::RichText::new(
                                            "Pick a preset in the Render tab on the left, or start right away below:",
                                        )
                                        .size(12.5)
                                        .color(egui::Color32::from_rgb(142, 168, 202)),
                                    );
                                    ui.add_space(10.0);
                                    if Self::icon_button(
                                        ui,
                                        UiIcon::Film,
                                        "Start Video Render (.mp4)",
                                        self.render_kind == StudioRenderKind::Animation,
                                        Some(egui::vec2(248.0, 34.0)),
                                    )
                                    .clicked()
                                    {
                                        self.active_tab = InspectorTab::Render;
                                        self.render_kind = StudioRenderKind::Animation;
                                        self.start_gui_mp4_render_with_ctx(Some(ctx.clone()));
                                    }
                                    ui.add_space(5.0);
                                    if Self::icon_button(
                                        ui,
                                        UiIcon::Camera,
                                        "Render High-Res Photo (F12)",
                                        self.render_kind == StudioRenderKind::StillFrame,
                                        Some(egui::vec2(248.0, 32.0)),
                                    )
                                    .clicked()
                                    {
                                        self.active_tab = InspectorTab::Render;
                                        self.render_kind = StudioRenderKind::StillFrame;
                                        self.start_gui_still_render(
                                            Some(ctx.clone()),
                                            self.render_auto_save_png,
                                        );
                                    }
                                });
                            });
                        });
                    }

                    if bg_is_running {
                        let eta_part = Self::format_eta(bg_done, bg_total, bg_elapsed)
                            .map(|s| format!(" • {}", s))
                            .unwrap_or_default();
                        let banner_txt = if !bg_phase.is_empty() {
                            format!("{}{}", bg_phase, eta_part)
                        } else if bg_kind == StudioRenderKind::Animation && bg_total > 0 {
                            format!(
                                "Rendering Video — Frame {} / {}{}",
                                bg_done, bg_total, eta_part
                            )
                        } else {
                            format!(
                                "Rendering {}×{} Studio Photo...",
                                self.video_width, self.video_height
                            )
                        };
                        let banner_rect = egui::Rect::from_center_size(
                            egui::pos2(rect.center().x, rect.top() + 26.0),
                            egui::vec2(460.0_f32.min(rect.width() - 24.0), 30.0),
                        );
                        let p = ui.painter().with_clip_rect(rect);
                        p.rect(
                            banner_rect,
                            egui::CornerRadius::same(6),
                            egui::Color32::from_rgba_unmultiplied(12, 18, 30, 228),
                            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(58, 134, 220)),
                            egui::StrokeKind::Inside,
                        );
                        p.text(
                            banner_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            banner_txt,
                            egui::FontId::proportional(12.5),
                            egui::Color32::from_rgb(168, 224, 255),
                        );
                    }
                    return;
                }

                let available_size = ui.available_size();
                let (rect, response) =
                    ui.allocate_exact_size(available_size, egui::Sense::click_and_drag());

                let is_shift_held = ctx.input(|i| i.modifiers.shift);
                let is_2d = self.is_2d;

                if is_2d {
                    self.camera.azimuth = 0.0;
                    self.camera.elevation = 0.0;
                }

                // Mouse Zoom
                let scroll_delta = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll_delta != 0.0 && response.hovered() {
                    self.camera.zoom(scroll_delta * 0.03);
                }

                // View and projection matrices
                let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
                let view = self.camera.view_matrix();
                let proj = self.camera.projection_matrix(aspect);
                let view_proj = proj * view;
                let fov_y = self.camera.fov_y;
                let eye_pos = self.camera.eye_position();

                // Handle Fluid Interaction (Left-Drag = Attract, Shift + Left-Drag = Repel)
                if response.dragged_by(egui::PointerButton::Primary) {
                    if let Some(pointer_pos) = response
                        .interact_pointer_pos()
                        .or_else(|| response.hover_pos())
                    {
                        let local_x = (pointer_pos.x - rect.min.x).clamp(0.0, rect.width());
                        let local_y = (pointer_pos.y - rect.min.y).clamp(0.0, rect.height());
                        let ray = unproject_mouse_ray(
                            glam::vec2(local_x, local_y),
                            glam::vec2(rect.width(), rect.height()),
                            view,
                            proj,
                        );
                        self.solver.mouse_ray = Some(ray);
                        self.solver.mouse_force_mode = if is_shift_held { -1 } else { 1 };
                    }
                } else {
                    self.solver.mouse_force_mode = 0;
                }

                // Camera Navigation: Middle-Drag = Pan, Right-Drag = Orbit (3D)
                if response.dragged_by(egui::PointerButton::Middle) {
                    let delta = response.drag_delta();
                    self.camera.pan(delta.x, delta.y);
                } else if !is_2d && response.dragged_by(egui::PointerButton::Secondary) {
                    let delta = response.drag_delta();
                    self.camera.rotate(delta.x * 0.006, -delta.y * 0.006);
                }

                let renderer_arc = self.renderer.clone();
                let bounds = self.solver.bounds;
                let chamber = self.solver.chamber.clone();
                let colliders = self.solver.colliders.clone();
                let faucets = self.solver.faucets.clone();
                let drains = self.solver.drains.clone();
                let show_grid = self.mesh_config.show_grid;
                let show_wireframe = self.mesh_config.show_wireframe;
                let exposure = self.mesh_config.exposure;
                let env_preset = self.mesh_config.env_preset;
                let light_dir = self.mesh_config.light_dir;
                let light_int = self.mesh_config.light_intensity;
                let amb_int = self.mesh_config.ambient_intensity;

                if self.render_mode == RenderMode::SurfaceMesh {
                    if self.mesh_dirty {
                        let t_m = Instant::now();
                        let mesh = SurfaceMesh::from_particles(
                            &self.solver.particles,
                            self.solver.config.particle_radius,
                            self.solver.config.smoothing_length,
                            &self.mesh_config,
                        );
                        self.cached_surface_mesh = Arc::new(mesh);
                        self.last_mesh_time_ms = t_m.elapsed().as_secs_f32() * 1000.0;
                        self.mesh_dirty = false;
                    }

                    let mesh_arc = Arc::clone(&self.cached_surface_mesh);
                    let mesh_cfg = self.mesh_config.clone();
                    let anim_time = self.sim_time;

                    let callback = egui_glow::CallbackFn::new(move |info, painter| {
                        if let Ok(mut guard) = renderer_arc.lock() {
                            if let Some(ref mut renderer) = *guard {
                                renderer.show_grid = show_grid;
                                renderer.show_wireframe = show_wireframe;
                                renderer.exposure = exposure;
                                renderer.env_preset = env_preset;
                                let gl = painter.gl();
                                let vp = info.viewport_in_pixels();
                                if vp.width_px > 0 && vp.height_px > 0 {
                                    unsafe {
                                        gl.viewport(
                                            vp.left_px,
                                            vp.from_bottom_px,
                                            vp.width_px,
                                            vp.height_px,
                                        );
                                    }

                                    renderer.render_surface_mesh(
                                        gl,
                                        view,
                                        proj,
                                        eye_pos,
                                        &mesh_arc,
                                        &mesh_cfg,
                                        anim_time,
                                        &chamber,
                                        bounds,
                                        &colliders,
                                        &faucets,
                                        &drains,
                                        is_2d,
                                    );

                                    unsafe {
                                        gl.viewport(
                                            0,
                                            0,
                                            info.screen_size_px[0] as i32,
                                            info.screen_size_px[1] as i32,
                                        );
                                    }
                                }
                            }
                        }
                    });

                    ui.painter().add(egui::PaintCallback {
                        rect,
                        callback: Arc::new(callback),
                    });
                } else {
                    // Populate persistent vertex buffer with parallel frustum culling
                    let enable_culling = self.enable_culling;
                    let color_mode = self.color_mode;
                    let color_map = self.color_map;
                    let particle_foam = self.particle_foam;
                    let rest_density = self.solver.config.rest_density;
                    let sim_bounds = self.solver.bounds;
                    self.render_vertices.clear();
                    self.render_vertices.par_extend(
                        self.solver.particles.par_iter().filter_map(|p| {
                            if !p.position.is_finite() {
                                return None;
                            }
                            if enable_culling {
                                let clip = view_proj
                                    * glam::Vec4::new(p.position.x, p.position.y, p.position.z, 1.0);
                                if clip.w <= 0.001 {
                                    return None;
                                }
                                let inv_w = 1.0 / clip.w;
                                let ndc_x = clip.x * inv_w;
                                let ndc_y = clip.y * inv_w;
                                let ndc_z = clip.z * inv_w;
                                let margin = 0.2;
                                if ndc_x < -1.0 - margin
                                    || ndc_x > 1.0 + margin
                                    || ndc_y < -1.0 - margin
                                    || ndc_y > 1.0 + margin
                                    || ndc_z < -1.0
                                    || ndc_z > 1.0
                                {
                                    return None;
                                }
                            }
                            let c = ColorMap::particle_color(
                                p,
                                color_mode,
                                color_map,
                                rest_density,
                                &sim_bounds,
                                particle_foam,
                            );
                            Some(ParticleVertex {
                                pos: p.position.to_array(),
                                color: [
                                    c[0] as f32 / 255.0,
                                    c[1] as f32 / 255.0,
                                    c[2] as f32 / 255.0,
                                    1.0,
                                ],
                            })
                        }),
                    );

                    self.drawn_particles_count = self.render_vertices.len();
                    let particles_snapshot = self.render_vertices.clone();
                    let lit_spheres = self.lit_spheres;
                    let particle_radius =
                        self.solver.config.particle_radius * self.display_size_factor;

                    let callback = egui_glow::CallbackFn::new(move |info, painter| {
                        if let Ok(mut guard) = renderer_arc.lock() {
                            if let Some(ref mut renderer) = *guard {
                                renderer.show_grid = show_grid;
                                renderer.show_wireframe = show_wireframe;
                                renderer.exposure = exposure;
                                renderer.env_preset = env_preset;
                                renderer.particle_light_dir = light_dir;
                                renderer.particle_light_intensity = light_int;
                                renderer.particle_ambient_intensity = amb_int;

                                let gl = painter.gl();
                                let vp = info.viewport_in_pixels();
                                if vp.width_px > 0 && vp.height_px > 0 {
                                    unsafe {
                                        gl.viewport(
                                            vp.left_px,
                                            vp.from_bottom_px,
                                            vp.width_px,
                                            vp.height_px,
                                        );
                                    }

                                    renderer.render(
                                        gl,
                                        view,
                                        proj,
                                        [vp.width_px as f32, vp.height_px as f32],
                                        particle_radius,
                                        fov_y,
                                        &particles_snapshot,
                                        lit_spheres,
                                        &chamber,
                                        bounds,
                                        &colliders,
                                        &faucets,
                                        &drains,
                                        is_2d,
                                    );

                                    unsafe {
                                        gl.viewport(
                                            0,
                                            0,
                                            info.screen_size_px[0] as i32,
                                            info.screen_size_px[1] as i32,
                                        );
                                    }
                                }
                            }
                        }
                    });

                    ui.painter().add(egui::PaintCallback {
                        rect,
                        callback: Arc::new(callback),
                    });
                }
            });

        // 5. Extreme Quality Confirmation Modal Pop-up
        if self.show_extreme_modal {
            egui::Window::new("Warning: Extreme Simulation Quality")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_max_width(450.0);
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let (wr, _) =
                            ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                        Self::paint_icon(
                            ui.painter(),
                            wr.center(),
                            UiIcon::Warning,
                            egui::Color32::from_rgb(255, 175, 45),
                        );
                        ui.heading("1,000,000+ Particles Workload");
                    });
                    ui.add_space(8.0);
                    ui.label(
                        "You are about to switch to the Extreme Quality preset, which spawns over \
                        1,000,000 physical fluid particles in a dense microscopic lattice.",
                    );
                    ui.add_space(6.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 130, 50),
                        "System Performance & Memory Advisory:",
                    );
                    ui.label(
                        "• Computing over 1M particles places a heavy load on your GPU and system RAM.\n\
                        • Your desktop window compositor may briefly stutter during buffer allocation.\n\
                        • Vulkan GPU compute is strongly recommended for real-time interaction.",
                    );
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            self.show_extreme_modal = false;
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let proceed_btn = egui::Button::new(
                                egui::RichText::new("Proceed to 1M+")
                                    .color(egui::Color32::BLACK)
                                    .strong(),
                            )
                            .fill(egui::Color32::from_rgb(255, 170, 0));

                            if ui.add(proceed_btn).clicked() {
                                self.show_extreme_modal = false;
                                self.set_quality(SimulationQuality::Extreme);
                            }
                        });
                    });
                });
        }

        // 6. Keyboard & Mouse Shortcuts Reference Modal
        if self.show_shortcuts_modal {
            let mut open = self.show_shortcuts_modal;
            egui::Window::new("Studio Keyboard & Mouse Controls")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_min_width(370.0);
                    egui::Grid::new("shortcuts_grid")
                        .num_columns(2)
                        .spacing([22.0, 6.0])
                        .striped(true)
                        .show(ui, |ui| {
                            for &(k, desc) in &[
                                ("Space", "Play / Pause simulation"),
                                (". (Period)", "Step single physics frame"),
                                ("R", "Reset current scene preset"),
                                ("C", "Reset camera framing"),
                                ("M", "Toggle Spheres / 3D Surface Mesh"),
                                ("G", "Toggle 3D studio floor grid"),
                                ("B", "Toggle chamber & obstacle wireframes"),
                                ("O", "Toggle 360° turntable auto-orbit"),
                                ("F12", "Render Studio Frame & open Render Result view"),
                                ("P", "Quick-save Studio PNG screenshot to exports/"),
                                ("Left / Right", "Step rendered animation frame in Render View"),
                                ("? / F1", "Show / hide this shortcuts window"),
                                ("Left-Drag", "Attract fluid toward cursor ray"),
                                ("Shift + Left-Drag", "Repel fluid away from cursor ray"),
                                ("Right-Drag", "Orbit 3D camera around target"),
                                ("Middle-Drag", "Pan camera target"),
                                ("Mouse Wheel", "Zoom camera in / out"),
                            ] {
                                ui.monospace(
                                    egui::RichText::new(k)
                                        .color(egui::Color32::from_rgb(138, 198, 255)),
                                );
                                ui.label(desc);
                                ui.end_row();
                            }
                        });
                });
            self.show_shortcuts_modal = open;
        }

        // 7. GitHub Release Self-Updater Notification & Progress Modal
        self.draw_updater_modal(ctx);
    }
}

impl LiquidApp {
    fn load_obj_obstacle(&mut self, path: &str) {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            self.status_message = "OBJ error: file path is empty".to_string();
            return;
        }
        let b_center = self.solver.bounds.center();
        let b_size = self.solver.bounds.size();
        let min_span = if self.is_2d {
            b_size.x.min(b_size.y)
        } else {
            b_size.x.min(b_size.y).min(b_size.z)
        };
        let initial_scale = (min_span * 0.28).clamp(0.15, 0.65);
        let initial_pos = Vec3::new(
            b_center.x,
            self.solver.bounds.min.y + b_size.y * 0.35,
            if self.is_2d { 0.0 } else { b_center.z },
        );

        match Collider::load_obj(trimmed, initial_pos, initial_scale) {
            Ok(mut col) => {
                let id = self.next_collider_id;
                self.next_collider_id += 1;
                col.id = id;
                let file_label = std::path::Path::new(trimmed)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(trimmed);
                let tri_count = if let ColliderShape::Mesh(ref m) = col.shape {
                    m.triangle_count()
                } else {
                    0
                };
                col.name = format!("OBJ #{} ({})", id, file_label);
                self.status_message = format!(
                    "Loaded OBJ '{}' ({} triangles, scaled to {:.2}m)",
                    file_label, tri_count, initial_scale
                );
                self.solver.colliders.push(col);
            }
            Err(e) => {
                self.status_message = format!("OBJ error: {}", e);
            }
        }
    }

    /// Spawn a non-blocking native OS Save File dialog for Render outputs:
    /// `kind == 0`: choose `.mp4` video output path
    /// `kind == 1`: choose `.png` auto-save photo path
    /// `kind == 2`: immediate "Save Rendered Image As..." `.png`
    fn spawn_render_file_dialog(&self, ctx: egui::Context, kind: u8) {
        if self.render_browse_pending.swap(true, Ordering::SeqCst) {
            return;
        }
        let pending = Arc::clone(&self.render_browse_pending);
        let slot = match kind {
            0 => Arc::clone(&self.render_mp4_browse_result),
            1 => Arc::clone(&self.render_png_browse_result),
            _ => Arc::clone(&self.render_save_result_png_slot),
        };
        let hint = match kind {
            0 => self.cli_mp4_output.trim().to_string(),
            1 => self.png_output_path.trim().to_string(),
            _ => format!("render_result_{:05}.png", self.exported_frames_count),
        };

        std::thread::spawn(move || {
            let mut dialog = if kind == 0 {
                rfd::FileDialog::new()
                    .set_title("Save Rendered Video As (.mp4)")
                    .add_filter("MP4 Video (*.mp4)", &["mp4", "MP4"])
            } else {
                rfd::FileDialog::new()
                    .set_title("Save Rendered Image As (.png)")
                    .add_filter("PNG Image (*.png)", &["png", "PNG"])
            };

            let hint_path = std::path::Path::new(&hint);
            if let Some(parent) = hint_path.parent() {
                if !parent.as_os_str().is_empty() && parent.is_dir() {
                    dialog = dialog.set_directory(parent);
                } else if let Ok(cwd) = std::env::current_dir() {
                    dialog = dialog.set_directory(cwd);
                }
            } else if let Ok(cwd) = std::env::current_dir() {
                dialog = dialog.set_directory(cwd);
            }

            let default_name = if kind == 0 {
                "simulation.mp4"
            } else {
                "render_photo.png"
            };
            let fname = hint_path
                .file_name()
                .and_then(|n| n.to_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(default_name);
            dialog = dialog.set_file_name(fname);

            if let Some(chosen) = dialog.save_file() {
                if let Ok(mut g) = slot.lock() {
                    *g = Some(chosen.to_string_lossy().into_owned());
                }
            }

            pending.store(false, Ordering::SeqCst);
            ctx.request_repaint();
        });
    }

    fn spawn_scene_file_dialog(&self, ctx: egui::Context, save: bool) {
        if self.scene_browse_pending.swap(true, Ordering::SeqCst) {
            return;
        }
        let pending = Arc::clone(&self.scene_browse_pending);
        let load_slot = Arc::clone(&self.scene_load_result);
        let save_slot = Arc::clone(&self.scene_save_result);
        let hint = self.scene_file_path.trim().to_string();

        std::thread::spawn(move || {
            let mut dialog = rfd::FileDialog::new()
                .set_title(if save {
                    "Save LiquidRust Studio Scene (.json)"
                } else {
                    "Open LiquidRust Studio Scene (.json)"
                })
                .add_filter("LiquidRust Scene (*.json)", &["json", "JSON"]);

            if let Ok(cwd) = std::env::current_dir() {
                dialog = dialog.set_directory(cwd);
            }
            if save {
                let fname = std::path::Path::new(&hint)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("scene.json");
                dialog = dialog.set_file_name(fname);
                if let Some(chosen) = dialog.save_file() {
                    if let Ok(mut g) = save_slot.lock() {
                        *g = Some(chosen.to_string_lossy().into_owned());
                    }
                }
            } else if let Some(chosen) = dialog.pick_file() {
                if let Ok(mut g) = load_slot.lock() {
                    *g = Some(chosen.to_string_lossy().into_owned());
                }
            }

            pending.store(false, Ordering::SeqCst);
            ctx.request_repaint();
        });
    }

    fn spawn_obj_file_browser(&self, ctx: egui::Context) {
        if self.obj_browse_pending.swap(true, Ordering::SeqCst) {
            return;
        }
        let pending = Arc::clone(&self.obj_browse_pending);
        let result_slot = Arc::clone(&self.obj_browse_result);
        let initial_hint = self.obj_file_path.trim().to_string();

        std::thread::spawn(move || {
            let mut dialog = rfd::FileDialog::new()
                .set_title("Select Wavefront .OBJ Obstacle Mesh")
                .add_filter("Wavefront OBJ Mesh (*.obj)", &["obj", "OBJ"])
                .add_filter("All Files", &["*"]);

            let hint_path = std::path::Path::new(&initial_hint);
            if let Some(parent) = hint_path.parent() {
                if !parent.as_os_str().is_empty() && parent.is_dir() {
                    dialog = dialog.set_directory(parent);
                } else if let Ok(cwd) = std::env::current_dir() {
                    dialog = dialog.set_directory(cwd);
                }
            } else if let Ok(cwd) = std::env::current_dir() {
                dialog = dialog.set_directory(cwd);
            }

            if let Some(chosen) = dialog.pick_file() {
                if let Ok(mut guard) = result_slot.lock() {
                    *guard = Some(chosen.to_string_lossy().into_owned());
                }
            }

            pending.store(false, Ordering::SeqCst);
            ctx.request_repaint();
        });
    }

    fn draw_updater_modal(&mut self, ctx: &egui::Context) {
        let snap = self.updater.snapshot();

        if let UpdatePhase::ReadyToRestart { ref target_exe, .. } = snap.phase {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            if let Err(e) = restart_into_updated_binary(target_exe) {
                self.status_message = e;
            }
            return;
        }

        if !snap.show_modal {
            return;
        }

        let title = match &snap.phase {
            UpdatePhase::UpdateAvailable(info) => {
                format!("Update Available — {}", info.tag_name)
            }
            UpdatePhase::Downloading { info, .. } => {
                format!("Downloading Update {}...", info.tag_name)
            }
            UpdatePhase::Installing(info) => {
                format!("Installing {}...", info.tag_name)
            }
            _ => "LiquidRust Software Update".to_string(),
        };

        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(420.0);
                ui.set_max_width(470.0);
                ui.add_space(4.0);

                match &snap.phase {
                    UpdatePhase::Idle | UpdatePhase::Checking => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(
                                egui::RichText::new(
                                    "Checking GitHub Releases (rappar222/liquid-rust)...",
                                )
                                .size(13.5),
                            );
                        });
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Close").clicked() {
                                    self.updater.close_modal();
                                }
                            });
                        });
                    }
                    UpdatePhase::UpToDate(info) => {
                        ui.colored_label(
                            egui::Color32::from_rgb(74, 222, 128),
                            egui::RichText::new(format!(
                                "You are running the latest version (v{}).",
                                CURRENT_VERSION
                            ))
                            .strong()
                            .size(14.0),
                        );
                        ui.add_space(4.0);
                        ui.weak(format!(
                            "Latest GitHub release: {} • Package: {} ({})",
                            info.tag_name,
                            info.asset_name,
                            format_bytes(info.asset_size)
                        ));
                        ui.add_space(12.0);
                        ui.separator();
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("Check Again").clicked() {
                                self.updater.spawn_check(Some(ctx.clone()), true);
                            }
                            if ui
                                .button("Reinstall & Restart")
                                .on_hover_text(
                                    "Download the latest asset for this release tag and restart",
                                )
                                .clicked()
                            {
                                self.updater
                                    .spawn_download_and_apply(info.clone(), Some(ctx.clone()));
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Close").clicked() {
                                    self.updater.close_modal();
                                }
                            });
                        });
                    }
                    UpdatePhase::UpdateAvailable(info) => {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "v{}  ->  {}",
                                    CURRENT_VERSION, info.tag_name
                                ))
                                .strong()
                                .size(15.0)
                                .color(egui::Color32::from_rgb(104, 214, 255)),
                            );
                            ui.weak(format!(
                                "• {} ({})",
                                info.asset_name,
                                format_bytes(info.asset_size)
                            ));
                        });
                        if !info.notes.trim().is_empty() {
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("Release Notes:").strong().size(12.0));
                            egui::Frame::NONE
                                .fill(egui::Color32::from_rgb(9, 13, 21))
                                .stroke(egui::Stroke::new(
                                    1.0_f32,
                                    egui::Color32::from_rgb(30, 41, 61),
                                ))
                                .corner_radius(egui::CornerRadius::same(6))
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    egui::ScrollArea::vertical()
                                        .max_height(160.0)
                                        .show(ui, |ui| {
                                            ui.label(
                                                egui::RichText::new(info.notes.trim()).size(12.0),
                                            );
                                        });
                                });
                        }
                        ui.add_space(12.0);
                        ui.separator();
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("Later").clicked() {
                                self.updater.close_modal();
                            }
                            if !info.html_url.is_empty() && ui.button("View on GitHub").clicked() {
                                ctx.open_url(egui::OpenUrl::new_tab(&info.html_url));
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let install_btn = egui::Button::new(
                                    egui::RichText::new("Update & Restart Now")
                                        .strong()
                                        .color(egui::Color32::from_rgb(10, 24, 18)),
                                )
                                .fill(egui::Color32::from_rgb(74, 222, 128));
                                if ui.add(install_btn).clicked() {
                                    self.updater
                                        .spawn_download_and_apply(info.clone(), Some(ctx.clone()));
                                }
                            });
                        });
                    }
                    UpdatePhase::Downloading {
                        info,
                        downloaded_bytes,
                        total_bytes,
                    } => {
                        let frac = if *total_bytes > 0 {
                            (*downloaded_bytes as f32 / *total_bytes as f32).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        ui.label(format!("Downloading {}...", info.asset_name));
                        ui.add_space(6.0);
                        ui.add(egui::ProgressBar::new(frac).show_percentage().text(format!(
                            "{} / {}",
                            format_bytes(*downloaded_bytes),
                            format_bytes(*total_bytes)
                        )));
                        ctx.request_repaint();
                    }
                    UpdatePhase::Installing(info) => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!(
                                "Extracting {} and replacing binary in-place...",
                                info.asset_name
                            ));
                        });
                        ctx.request_repaint();
                    }
                    UpdatePhase::ReadyToRestart { .. } => {
                        ui.label("Restarting into updated LiquidRust...");
                    }
                    UpdatePhase::Error(err) => {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 115, 115),
                            format!("Update error: {}", err),
                        );
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            if ui.button("Retry").clicked() {
                                self.updater.spawn_check(Some(ctx.clone()), true);
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Close").clicked() {
                                    self.updater.close_modal();
                                }
                            });
                        });
                    }
                }
            });
    }
}
