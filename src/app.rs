use crate::export::{Mp4Encoder, PlyExporter, PngExporter, SoftwareRenderer};
use crate::gpu::GpuSphSolver;
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
use crate::sph::{ChamberShape, Collider, ColliderShape, CubeFace, Drain, Faucet, FaucetShape};
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
    RenderCli,
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

/// Shared progress state for non-blocking background MP4 video rendering initiated from the GUI.
pub struct BgRenderState {
    pub is_running: bool,
    pub completed_frames: usize,
    pub total_frames: usize,
    pub elapsed_secs: f32,
    pub finished_message: Option<String>,
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

    // Inspector & Export / CLI Generator
    pub active_tab: InspectorTab,
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
            gui_render_quality: CliRenderQuality::Standard,
            video_width: 1280,
            video_height: 720,
            bg_render: Arc::new(Mutex::new(BgRenderState {
                is_running: false,
                completed_frames: 0,
                total_frames: 0,
                elapsed_secs: 0.0,
                finished_message: None,
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
            FluidPhysicsPreset::Mercury => ColorMapType::Ice,
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

    pub fn export_current_png(&mut self) {
        let dir = std::path::Path::new(&self.export_path);
        if let Err(e) = std::fs::create_dir_all(dir) {
            self.status_message = format!("PNG export error: {}", e);
            return;
        }
        let path = dir.join(format!("fluid_frame_{:05}.png", self.exported_frames_count));
        let width = self.video_width.max(320);
        let height = self.video_height.max(240);

        let mut sw = SoftwareRenderer::new_with_quality(width, height, self.gui_render_quality);
        sw.enable_culling = self.enable_culling;
        sw.lit_spheres = self.lit_spheres;
        sw.particle_foam = self.particle_foam;
        sw.env_preset = self.mesh_config.env_preset;
        sw.exposure = self.mesh_config.exposure;
        sw.show_grid = self.mesh_config.show_grid;
        sw.show_wireframe = self.mesh_config.show_wireframe;
        sw.light_dir = self.mesh_config.light_dir;
        sw.light_intensity = self.mesh_config.light_intensity;
        sw.ambient_intensity = self.mesh_config.ambient_intensity;

        match self.render_mode {
            RenderMode::Particles => {
                sw.render_particles_studio(
                    &self.solver.particles,
                    self.solver.config.particle_radius * self.display_size_factor,
                    &self.solver.chamber,
                    self.solver.bounds,
                    &self.solver.colliders,
                    &self.camera,
                    self.color_mode,
                    self.color_map,
                    self.solver.config.rest_density,
                    self.is_2d,
                );
            }
            RenderMode::SurfaceMesh => {
                let mut mcfg = self.mesh_config.clone();
                self.gui_render_quality
                    .apply_to_mesh_config(&mut mcfg, false, false);
                let mesh = SurfaceMesh::from_particles(
                    &self.solver.particles,
                    self.solver.config.particle_radius,
                    self.solver.config.smoothing_length,
                    &mcfg,
                );
                sw.render_surface_mesh_frame(
                    &mesh,
                    &mcfg,
                    &self.camera,
                    &self.solver.chamber,
                    self.solver.bounds,
                    &self.solver.colliders,
                    self.sim_time,
                    self.is_2d,
                );
            }
        }

        match PngExporter::save_rgb(&path, width, height, &sw.rgb_buffer) {
            Ok(_) => {
                self.status_message =
                    format!("Saved {}x{} PNG to {}", width, height, path.display());
                self.exported_frames_count += 1;
            }
            Err(e) => {
                self.status_message = format!("PNG export error: {}", e);
            }
        }
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

    /// Launch a non-blocking studio MP4 render directly from the GUI using current scene & surface mesh settings.
    pub fn start_gui_mp4_render(&mut self) {
        if let Ok(guard) = self.bg_render.lock() {
            if guard.is_running {
                return;
            }
        }

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

        // Clone an independent simulation snapshot starting cleanly from the current preset & scene objects
        let active_cfg = self.solver.config.clone();
        let mut render_solver = SphSolver::new(self.solver.bounds, active_cfg.clone());
        render_solver.set_2d(self.is_2d);
        render_solver.set_chamber(self.solver.chamber.clone());
        SceneGenerator::setup_scene(&mut render_solver, self.current_preset);
        render_solver.config = active_cfg;
        render_solver.colliders = self.solver.colliders.clone();
        render_solver.faucets = self.solver.faucets.clone();
        render_solver.drains = self.solver.drains.clone();

        let render_mode = self.render_mode;
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

        self.bg_render_cancel.store(false, Ordering::SeqCst);
        let cancel_flag = Arc::clone(&self.bg_render_cancel);
        let state_arc = Arc::clone(&self.bg_render);

        if let Ok(mut guard) = state_arc.lock() {
            guard.is_running = true;
            guard.completed_frames = 0;
            guard.total_frames = total_frames;
            guard.elapsed_secs = 0.0;
            guard.finished_message = None;
        }
        self.status_message = format!("Rendering {} ({} frames)...", out_path, total_frames);

        std::thread::spawn(move || {
            let t_start = Instant::now();
            let mut encoder = match Mp4Encoder::new_with_encoding(
                &out_path,
                width,
                height,
                fps,
                rq.ffmpeg_preset(),
                rq.ffmpeg_crf(),
            ) {
                Ok(enc) => enc,
                Err(e) => {
                    if let Ok(mut g) = state_arc.lock() {
                        g.is_running = false;
                        g.finished_message = Some(format!("MP4 error: {}", e));
                    }
                    return;
                }
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

            let mut anim_t = 0.0f32;
            let mut done = 0usize;

            for _frame in 0..total_frames {
                if cancel_flag.load(Ordering::SeqCst) {
                    break;
                }

                if let Some(ref mut gpu) = gpu_opt {
                    let prev_n = render_solver.particles.len();
                    render_solver.emit_faucets(dt);
                    if render_solver.particles.len() != prev_n {
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
                    gpu.read_particles(&mut render_solver.particles);

                    let has_mesh_col = render_solver
                        .colliders
                        .iter()
                        .any(|c| c.enabled && matches!(c.shape, ColliderShape::Mesh(_)));
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
                                        p_hit |=
                                            col.resolve(&mut p.position, &mut p.velocity, radius);
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
                    render_solver.step(dt);
                }
                anim_t += dt;

                match render_mode {
                    RenderMode::Particles => {
                        sw.render_particles_studio(
                            &render_solver.particles,
                            render_solver.config.particle_radius * display_size_factor,
                            &render_solver.chamber,
                            render_solver.bounds,
                            &render_solver.colliders,
                            &camera,
                            color_mode,
                            color_map,
                            render_solver.config.rest_density,
                            is_2d,
                        );
                    }
                    RenderMode::SurfaceMesh => {
                        let mesh = SurfaceMesh::from_particles(
                            &render_solver.particles,
                            render_solver.config.particle_radius,
                            render_solver.config.smoothing_length,
                            &mesh_cfg,
                        );
                        sw.render_surface_mesh_frame(
                            &mesh,
                            &mesh_cfg,
                            &camera,
                            &render_solver.chamber,
                            render_solver.bounds,
                            &render_solver.colliders,
                            anim_t,
                            is_2d,
                        );
                    }
                }

                camera.azimuth += orbit_rad_per_sec * wall_frame_dt;

                if let Err(e) = encoder.write_frame(&sw.rgb_buffer) {
                    if let Ok(mut g) = state_arc.lock() {
                        g.is_running = false;
                        g.finished_message = Some(format!("Encode error: {}", e));
                    }
                    return;
                }

                done += 1;
                if let Ok(mut g) = state_arc.lock() {
                    g.completed_frames = done;
                    g.elapsed_secs = t_start.elapsed().as_secs_f32();
                }
            }

            let finish_res = encoder.finish();
            if let Ok(mut g) = state_arc.lock() {
                g.is_running = false;
                g.completed_frames = done;
                g.elapsed_secs = t_start.elapsed().as_secs_f32();
                g.finished_message = Some(match finish_res {
                    Ok(_) => format!(
                        "Saved {} ({} frames in {:.1}s)",
                        out_path, done, g.elapsed_secs
                    ),
                    Err(e) => format!("FFmpeg finalize error: {}", e),
                });
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

        // 2. Direct Studio MP4 Video Renderer
        Self::section_card(ui, "Studio MP4 Video Renderer", |ui| {
            ui.horizontal(|ui| {
                ui.label("Mode:");
                if Self::icon_button(
                    ui,
                    UiIcon::Spheres,
                    "Spheres",
                    self.render_mode == RenderMode::Particles,
                    None,
                )
                .clicked()
                {
                    self.render_mode = RenderMode::Particles;
                }
                if Self::icon_button(
                    ui,
                    UiIcon::Mesh,
                    "Surface Mesh",
                    self.render_mode == RenderMode::SurfaceMesh,
                    None,
                )
                .clicked()
                {
                    self.render_mode = RenderMode::SurfaceMesh;
                    self.mesh_dirty = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Output MP4:");
                ui.text_edit_singleline(&mut self.cli_mp4_output);
            });

            ui.horizontal(|ui| {
                ui.label("Resolution:");
                for &(w, h, lbl) in &[
                    (1280usize, 720usize, "720p"),
                    (1920usize, 1080usize, "1080p"),
                    (2560usize, 1440usize, "1440p"),
                    (3840usize, 2160usize, "4K"),
                ] {
                    if ui
                        .selectable_label(self.video_width == w && self.video_height == h, lbl)
                        .clicked()
                    {
                        self.video_width = w;
                        self.video_height = h;
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Optics Tier:");
                for &rq in CliRenderQuality::ALL {
                    let short = match rq {
                        CliRenderQuality::Standard => "Standard",
                        CliRenderQuality::Studio => "Studio 2x",
                        CliRenderQuality::Cinema => "Cinema 3x",
                        CliRenderQuality::Master => "Master 4x",
                    };
                    if ui
                        .selectable_label(self.gui_render_quality == rq, short)
                        .on_hover_text(rq.name())
                        .clicked()
                    {
                        self.gui_render_quality = rq;
                    }
                }
            });

            ui.add(
                egui::Slider::new(&mut self.cli_duration, 0.5..=30.0)
                    .text("Duration (s)")
                    .fixed_decimals(1),
            );

            ui.horizontal(|ui| {
                ui.label("FPS:");
                for &f in &[24u32, 30, 60, 120] {
                    if ui
                        .selectable_label(self.cli_fps == f, format!("{}", f))
                        .clicked()
                    {
                        self.cli_fps = f;
                    }
                }
            });

            ui.add_space(4.0);

            let (is_running, done, total, elapsed, finished_msg) = {
                if let Ok(mut g) = self.bg_render.lock() {
                    let msg = g.finished_message.take();
                    (
                        g.is_running,
                        g.completed_frames,
                        g.total_frames,
                        g.elapsed_secs,
                        msg,
                    )
                } else {
                    (false, 0, 0, 0.0, None)
                }
            };
            if let Some(msg) = finished_msg {
                self.status_message = msg;
            }

            if is_running {
                let frac = if total > 0 {
                    done as f32 / total as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .show_percentage()
                        .text(format!("Frame {}/{} ({:.1}s)", done, total, elapsed)),
                );
                if ui.button("Stop & Finalize Video").clicked() {
                    self.bg_render_cancel.store(true, Ordering::SeqCst);
                }
                ctx.request_repaint();
            } else if Self::icon_button(
                ui,
                UiIcon::Film,
                "Render MP4 Video Now",
                true,
                Some(egui::vec2(ui.available_width(), 30.0)),
            )
            .clicked()
            {
                self.start_gui_mp4_render();
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

        // Studio keyboard shortcuts (only when not typing into a text input widget)
        if !ctx.wants_keyboard_input() {
            let mut key_step = false;
            let mut key_reset = false;
            let mut key_cam = false;
            let mut key_mesh = false;
            let mut key_grid = false;
            let mut key_wire = false;
            let mut key_orbit = false;
            let mut key_png = false;
            let mut key_help = false;

            ctx.input(|i| {
                if i.key_pressed(egui::Key::Space) {
                    self.is_playing = !self.is_playing;
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
                if i.key_pressed(egui::Key::P) || i.key_pressed(egui::Key::F12) {
                    key_png = true;
                }
                if i.key_pressed(egui::Key::F1)
                    || (i.key_pressed(egui::Key::Slash) && i.modifiers.shift)
                {
                    key_help = true;
                }
            });

            if key_step {
                let dt = self.fixed_dt * self.sim_speed;
                self.step_simulation_once(dt);
                ctx.request_repaint();
            }
            if key_reset {
                self.load_preset(self.current_preset);
            }
            if key_cam {
                self.reset_camera_framing();
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
                            "Space: Play • R: Reset • M: Mesh • P: PNG • ?: Help • L-Drag: Pull • Shift+L-Drag: Push"
                        } else {
                            "Space: Play • R: Reset • M: Mesh • O: Orbit • P: PNG • ?: Help • R-Drag: Orbit"
                        });
                    });
                });
            });

        // 3. Left Studio Inspector Panel with 4 Organized Tabs
        egui::SidePanel::left("studio_inspector")
            .default_width(332.0)
            .min_width(310.0)
            .max_width(430.0)
            .show(ctx, |ui| {
                ui.add_space(3.0);
                ui.horizontal(|ui| {
                    let tab_w = ((ui.available_width() - 21.0) / 4.0).max(68.0);
                    for &(tab, icon, label) in &[
                        (InspectorTab::Scene, UiIcon::Wave, "Scene"),
                        (InspectorTab::Physics, UiIcon::Sliders, "Physics"),
                        (InspectorTab::Surface, UiIcon::Mesh, "Visuals"),
                        (InspectorTab::RenderCli, UiIcon::Film, "Export"),
                    ] {
                        if Self::icon_button(
                            ui,
                            icon,
                            label,
                            self.active_tab == tab,
                            Some(egui::vec2(tab_w, 26.0)),
                        )
                        .clicked()
                        {
                            self.active_tab = tab;
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
                            InspectorTab::RenderCli => self.draw_render_cli_tab(ctx, ui),
                        }
                        ui.add_space(10.0);
                    });
            });

        // 4. Central Viewport: Custom OpenGL 3D Render (Particles OR Real-Time Surface Mesh)
        let bg_rgb = self.mesh_config.env_preset.viewport_bg_rgb8();
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE.fill(egui::Color32::from_rgb(bg_rgb[0], bg_rgb[1], bg_rgb[2])),
            )
            .show(ctx, |ui| {
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
                                ("P / F12", "Capture Studio PNG screenshot"),
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
