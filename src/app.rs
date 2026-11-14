use crate::export::{Mp4Encoder, PlyExporter, SoftwareRenderer};
use crate::gpu::GpuSphSolver;
use crate::math::{unproject_mouse_ray, Aabb};
use crate::render::{
    Camera, ColorMap, ColorMapType, ColorMode, FluidMaterialPreset, GlRenderer, MeshTextureMode,
    ParticleVertex, RenderMode, SurfaceMesh, SurfaceMeshConfig,
};
use crate::sph::scene::{SceneGenerator, ScenePreset};
use crate::sph::solver::{
    compute_lattice_rest_density, SimulationQuality, SolverConfig, SphSolver,
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
    Headless,
    PlySequence,
}

impl CliCommandMode {
    pub const ALL: &'static [CliCommandMode] = &[
        CliCommandMode::Gui,
        CliCommandMode::Mp4Video,
        CliCommandMode::Headless,
        CliCommandMode::PlySequence,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            CliCommandMode::Gui => "Interactive GUI",
            CliCommandMode::Mp4Video => "MP4 Video",
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
    pub is_playing: bool,
    pub fixed_dt: f32,
    pub sim_speed: f32,
    pub current_preset: ScenePreset,
    pub current_quality: SimulationQuality,

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
    pub frame_counter: usize,
    pub current_fps: f32,
    pub last_step_time_ms: f32,

    // Inspector & Export / CLI Generator
    pub active_tab: InspectorTab,
    pub theme_applied: bool,
    pub export_path: String,
    pub export_sequence: bool,
    pub exported_frames_count: usize,
    pub status_message: String,
    pub show_extreme_modal: bool,
    pub obj_file_path: String,
    pub obj_browse_pending: Arc<AtomicBool>,
    pub obj_browse_result: Arc<Mutex<Option<String>>>,
    pub next_collider_id: u32,
    pub next_faucet_id: u32,
    pub next_drain_id: u32,
    pub is_2d: bool,
    pub cli_mode: CliCommandMode,
    pub cli_duration: f32,
    pub cli_fps: u32,
    pub cli_mp4_output: String,
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

        Self {
            solver,
            gpu_solver,
            use_gpu,
            gpu_device_name,
            camera: Camera::default(),
            is_playing,
            fixed_dt: 0.016,
            sim_speed: 1.0,
            current_preset: preset,
            current_quality: SimulationQuality::Medium,
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
            enable_culling: true,
            drawn_particles_count: 0,
            color_mode: ColorMode::Velocity,
            color_map: ColorMapType::Ocean,
            display_size_factor: 1.25,
            sphere_obstacle_pos: Vec3::new(0.0, 0.45, 0.0),
            sphere_obstacle_radius: 0.22,
            sphere_obstacle_enabled: false,
            render_vertices: Vec::with_capacity(init_cap),
            last_frame_time: Instant::now(),
            frame_counter: 0,
            current_fps: 60.0,
            last_step_time_ms: 0.0,
            active_tab: InspectorTab::Scene,
            theme_applied: false,
            export_path: "exports".to_string(),
            export_sequence: false,
            exported_frames_count: 0,
            status_message: "Ready".to_string(),
            show_extreme_modal: false,
            obj_file_path: "model.obj".to_string(),
            obj_browse_pending: Arc::new(AtomicBool::new(false)),
            obj_browse_result: Arc::new(Mutex::new(None)),
            next_collider_id: 1,
            next_faucet_id: 1,
            next_drain_id: 1,
            is_2d: false,
            cli_mode: CliCommandMode::Gui,
            cli_duration: 5.0,
            cli_fps: 60,
            cli_mp4_output: "simulation.mp4".to_string(),
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
        visuals.panel_fill = egui::Color32::from_rgb(11, 15, 23);
        visuals.window_fill = egui::Color32::from_rgb(14, 19, 30);
        visuals.extreme_bg_color = egui::Color32::from_rgb(7, 10, 16);
        visuals.faint_bg_color = egui::Color32::from_rgb(17, 23, 36);
        visuals.code_bg_color = egui::Color32::from_rgb(9, 13, 21);

        let border_subtle = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(34, 46, 68));
        let border_hover = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(66, 138, 225));
        let accent_fill = egui::Color32::from_rgb(22, 108, 196);
        let accent_stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(78, 178, 255));

        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(15, 20, 31);
        visuals.widgets.noninteractive.bg_stroke = border_subtle;
        visuals.widgets.noninteractive.fg_stroke =
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(206, 216, 232));
        visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(6);

        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(22, 30, 46);
        visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(19, 26, 40);
        visuals.widgets.inactive.bg_stroke = border_subtle;
        visuals.widgets.inactive.fg_stroke =
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(222, 230, 244));
        visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(6);

        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(31, 43, 66);
        visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(28, 39, 60);
        visuals.widgets.hovered.bg_stroke = border_hover;
        visuals.widgets.hovered.fg_stroke =
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(245, 250, 255));
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
        style.spacing.item_spacing = egui::vec2(7.0, 6.0);
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

    pub fn set_2d(&mut self, is_2d: bool) {
        self.is_2d = is_2d;
        self.solver.set_2d(is_2d);
        if is_2d {
            self.camera.azimuth = 0.0;
            self.camera.elevation = 0.0;
            self.camera.target = Vec3::new(0.0, 0.6, 0.0);
            self.camera.distance = 2.4;
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
        if let Some(ref mut gpu) = self.gpu_solver {
            gpu.init_buffers(
                self.solver.bounds,
                &self.solver.config,
                &self.solver.particles,
            );
        }
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

    fn step_simulation_once(&mut self, effective_dt: f32) {
        let start = Instant::now();

        if self.use_gpu && self.gpu_solver.is_some() {
            let prev_len = self.solver.particles.len();
            self.solver.emit_faucets(effective_dt);
            let emitted = self.solver.particles.len() != prev_len;

            if let Some(ref mut gpu) = self.gpu_solver {
                if emitted {
                    // Incremental slice upload instead of reallocating all GPU buffers
                    gpu.sync_particles(
                        self.solver.bounds,
                        &self.solver.config,
                        &self.solver.particles,
                        false,
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
                            p.position = p.position.clamp(b_min, b_max);
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
                    gpu.sync_particles(
                        self.solver.bounds,
                        &self.solver.config,
                        &self.solver.particles,
                        true,
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
        let width = self.video_width.max(320);
        let height = self.video_height.max(240);

        // Clone an independent simulation snapshot starting cleanly from the current preset & scene objects
        let mut render_solver = SphSolver::new(self.solver.bounds, self.solver.config.clone());
        render_solver.set_2d(self.is_2d);
        render_solver.set_chamber(self.solver.chamber.clone());
        SceneGenerator::setup_scene(&mut render_solver, self.current_preset);
        render_solver.colliders = self.solver.colliders.clone();
        render_solver.faucets = self.solver.faucets.clone();
        render_solver.drains = self.solver.drains.clone();

        let render_mode = self.render_mode;
        let mesh_cfg = self.mesh_config.clone();
        let camera = self.camera.clone();
        let color_mode = self.color_mode;
        let color_map = self.color_map;
        let display_size_factor = self.display_size_factor;
        let enable_culling = self.enable_culling;
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
            let mut encoder = match Mp4Encoder::new(&out_path, width, height, fps) {
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

            let mut sw = SoftwareRenderer::new(width, height);
            sw.enable_culling = enable_culling;
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
                            false,
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
                    let drained = render_solver.drain_particles();
                    if drained > 0 {
                        gpu.sync_particles(
                            render_solver.bounds,
                            &render_solver.config,
                            &render_solver.particles,
                            true,
                        );
                    }
                } else {
                    render_solver.step(dt);
                }
                anim_t += dt;

                match render_mode {
                    RenderMode::Particles => {
                        sw.render_frame(
                            &render_solver.particles,
                            render_solver.config.particle_radius * display_size_factor,
                            &render_solver.chamber,
                            render_solver.bounds,
                            &camera,
                            color_mode,
                            color_map,
                            render_solver.config.rest_density,
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

        // Surface Meshing & Optical Material/Texture/Lighting flags (when SurfaceMesh mode is selected)
        if self.render_mode == RenderMode::SurfaceMesh {
            parts.push("--surface-mesh".to_string());
            let mc = &self.mesh_config;
            let def = SurfaceMeshConfig::default();

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
            if !mc.cast_shadows {
                parts.push("--no-shadows".to_string());
            }
        }

        // Non-default physics parameters
        let def_phys = SolverConfig::default();
        if (self.solver.config.vorticity - def_phys.vorticity).abs() > 0.005 {
            parts.push(format!("--vorticity {:.3}", self.solver.config.vorticity));
        }
        if (self.solver.config.surface_tension - def_phys.surface_tension).abs() > 0.005 {
            parts.push(format!(
                "--surface-tension {:.3}",
                self.solver.config.surface_tension
            ));
        }

        // Non-default camera framing for MP4 renders
        if self.cli_mode == CliCommandMode::Mp4Video && !self.is_2d {
            let def_cam = Camera::default();
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

        // Chamber shape, size & scale
        if self.solver.chamber.shape != ChamberShape::Box {
            let shape_str = match self.solver.chamber.shape {
                ChamberShape::Bathtub => "bathtub",
                ChamberShape::Sphere => "sphere",
                ChamberShape::Box => "box",
            };
            parts.push(format!("--chamber-shape {}", shape_str));
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
            match &col.shape {
                ColliderShape::Sphere { center, radius } => {
                    parts.push(format!(
                        "--obstacle \"sphere:{:.2},{:.2},{:.2}:{:.2}\"",
                        center.x, center.y, center.z, radius
                    ));
                }
                ColliderShape::Box { aabb } => {
                    let c = aabb.center();
                    let s = aabb.size();
                    parts.push(format!(
                        "--obstacle \"box:{:.2},{:.2},{:.2}:{:.2},{:.2},{:.2}\"",
                        c.x, c.y, c.z, s.x, s.y, s.z
                    ));
                }
                ColliderShape::Cylinder {
                    center,
                    radius,
                    height,
                } => {
                    parts.push(format!(
                        "--obstacle \"cylinder:{:.2},{:.2},{:.2}:{:.2}:{:.2}\"",
                        center.x, center.y, center.z, radius, height
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
                CubeFace::Ceiling => parts.push(format!("--drain \"ceiling:{:.2}\"", d.radius)),
                CubeFace::Left => parts.push(format!("--drain \"left:{:.2}\"", d.radius)),
                CubeFace::Right => parts.push(format!("--drain \"right:{:.2}\"", d.radius)),
                CubeFace::Back => parts.push(format!("--drain \"back:{:.2}\"", d.radius)),
                CubeFace::Front => parts.push(format!("--drain \"front:{:.2}\"", d.radius)),
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
        let t = match self.color_mode {
            ColorMode::Velocity => (p.speed() / 2.4).clamp(0.0, 1.0),
            ColorMode::Pressure => {
                (p.pressure / (self.solver.config.rest_density * 0.4)).clamp(0.0, 1.0)
            }
            ColorMode::Density => {
                let dev = (p.density - self.solver.config.rest_density).abs()
                    / self.solver.config.rest_density;
                (dev * 4.0).clamp(0.0, 1.0)
            }
            ColorMode::Height => {
                let y = p.position.y;
                let b = &self.solver.bounds;
                ((y - b.min.y) / (b.max.y - b.min.y)).clamp(0.0, 1.0)
            }
            ColorMode::KineticEnergy => (p.kinetic_energy() / 4.0).clamp(0.0, 1.0),
            ColorMode::SingleColor => return [30, 140, 225],
        };
        ColorMap::sample(self.color_map, t)
    }

    fn draw_telemetry_pill(ui: &mut egui::Ui, text: String, color: egui::Color32) {
        egui::Frame::NONE
            .fill(egui::Color32::from_rgb(16, 22, 34))
            .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(34, 46, 68)))
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
            .fill(egui::Color32::from_rgb(15, 21, 33))
            .stroke(egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(30, 41, 61)))
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(
                    egui::RichText::new(title)
                        .strong()
                        .size(12.5)
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
        // 0. 3D / 2D Simulation Space Switcher
        Self::section_card(ui, "Simulation Space", |ui| {
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 6.0) * 0.5;
                let btn_3d = egui::Button::new(
                    egui::RichText::new("🌐 3D Volumetric").strong().color(if !self.is_2d {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_rgb(175, 190, 210)
                    }),
                )
                .fill(if !self.is_2d {
                    egui::Color32::from_rgb(14, 116, 184)
                } else {
                    egui::Color32::from_rgb(30, 38, 50)
                });
                if ui.add_sized([w, 26.0], btn_3d).clicked() && self.is_2d {
                    self.set_2d(false);
                }

                let btn_2d = egui::Button::new(
                    egui::RichText::new("▭ 2D Planar").strong().color(if self.is_2d {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_rgb(175, 190, 210)
                    }),
                )
                .fill(if self.is_2d {
                    egui::Color32::from_rgb(14, 116, 184)
                } else {
                    egui::Color32::from_rgb(30, 38, 50)
                });
                if ui.add_sized([w, 26.0], btn_2d).clicked() && !self.is_2d {
                    self.set_2d(true);
                }
            });
        });

        ui.add_space(6.0);

        // 1. Scene Presets
        Self::section_card(ui, "Scene Presets", |ui| {
            egui::Grid::new("preset_grid")
                .num_columns(2)
                .spacing([6.0, 5.0])
                .show(ui, |ui| {
                    for (i, &preset) in ScenePreset::ALL.iter().enumerate() {
                        let is_cur = self.current_preset == preset;
                        if ui
                            .add_sized(
                                [142.0, 24.0],
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
        Self::section_card(ui, "Particle Resolution & Lattice", |ui| {
            for &q in SimulationQuality::ALL {
                let is_cur = self.current_quality == q;
                let label_text = if q == SimulationQuality::Extreme {
                    egui::RichText::new(q.name_for(self.is_2d))
                        .color(egui::Color32::from_rgb(255, 145, 65))
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

            ui.separator();
            let mut changed = false;
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.solver.config.particle_radius, 0.010..=0.040)
                        .text("Radius (m)"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.solver.config.spacing, 0.018..=0.065)
                        .text("Spacing (m)"),
                )
                .changed();

            if changed {
                self.current_quality = SimulationQuality::Custom;
                self.solver.config.smoothing_length = self.solver.config.particle_radius * 3.1;
                self.solver.config.rest_density = compute_lattice_rest_density(
                    self.solver.config.smoothing_length,
                    self.solver.config.spacing,
                );
            }

            if ui.button("⚡ Respawn Lattice").clicked() {
                let h = self.solver.config.smoothing_length;
                self.solver.set_smoothing_length(h);
                self.load_preset(self.current_preset);
            }
        });

        ui.add_space(6.0);

        // 3. Simulation Chamber & Bounds
        Self::section_card(ui, "Container Chamber & Scale", |ui| {
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

            let mut scale_changed = false;
            let mut uniform_s = (self.solver.chamber.scale.x
                + self.solver.chamber.scale.y
                + self.solver.chamber.scale.z)
                / 3.0;
            if ui
                .add(egui::Slider::new(&mut uniform_s, 0.4..=3.0).text("Uniform"))
                .changed()
            {
                self.solver.chamber.scale = Vec3::splat(uniform_s);
                scale_changed = true;
            }
            scale_changed |= ui
                .add(egui::Slider::new(&mut self.solver.chamber.scale.x, 0.3..=3.0).text("Length X"))
                .changed();
            scale_changed |= ui
                .add(egui::Slider::new(&mut self.solver.chamber.scale.y, 0.3..=3.0).text("Height Y"))
                .changed();
            if !self.is_2d {
                scale_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.solver.chamber.scale.z, 0.3..=3.0)
                            .text("Depth Z"),
                    )
                    .changed();
            }

            if self.solver.chamber.shape == ChamberShape::Bathtub {
                ui.add(
                    egui::Slider::new(&mut self.solver.chamber.tub_slope_ratio, 0.1..=0.45)
                        .text("Backrest Slope"),
                );
                if !self.is_2d {
                    ui.add(
                        egui::Slider::new(&mut self.solver.chamber.tub_fillet_ratio, 0.05..=0.35)
                            .text("Bottom Fillet"),
                    );
                }
            }

            if ui.small_button("↺ Reset 1.0x").clicked() {
                self.solver.chamber.scale = Vec3::ONE;
                scale_changed = true;
            }

            if scale_changed {
                self.solver.bounds = self.solver.chamber.compute_bounds();
            }

            let sz = self.solver.bounds.size();
            ui.weak(if self.is_2d {
                format!("Bounds: {:.2} × {:.2} m", sz.x, sz.y)
            } else {
                format!("Bounds: {:.2} × {:.2} × {:.2} m", sz.x, sz.y, sz.z)
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
                        let mut c = Collider::cylinder(Vec3::new(0.0, 0.45, 0.0), 0.15, 0.4);
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
                        egui::TextEdit::singleline(&mut self.obj_file_path).desired_width(126.0),
                    );
                    let enter_pressed =
                        text_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                    let browsing = self.obj_browse_pending.load(Ordering::Relaxed);
                    if ui
                        .add_enabled(!browsing, egui::Button::new("📁 Browse…"))
                        .on_hover_text("Browse your computer for a Wavefront .OBJ 3D mesh")
                        .clicked()
                    {
                        self.spawn_obj_file_browser(ui.ctx().clone());
                    }

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
                                .small_button("↺")
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
                                        .text("Scale (Radius)")
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

                                // Move sliders (X, Y, Z)
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

                                // Scale slider (normalized: 1.0 = 1.0 meter bounding size)
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut mesh.scale, 0.04..=2.20)
                                            .text("Scale")
                                            .suffix(" m")
                                            .clamping(egui::SliderClamping::Edits),
                                    )
                                    .changed();

                                // Rotate sliders (Pitch X, Yaw Y, Roll Z in degrees)
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
                            ui.add(egui::DragValue::new(&mut faucet.position.x).speed(0.01).prefix("X:"));
                            ui.add(egui::DragValue::new(&mut faucet.position.y).speed(0.01).prefix("Y:"));
                            if !self.is_2d {
                                ui.add(egui::DragValue::new(&mut faucet.position.z).speed(0.01).prefix("Z:"));
                            } else {
                                faucet.position.z = 0.0;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.weak("Vel:");
                            ui.add(egui::DragValue::new(&mut faucet.direction.x).speed(0.05).prefix("Vx:"));
                            ui.add(egui::DragValue::new(&mut faucet.direction.y).speed(0.05).prefix("Vy:"));
                            if !self.is_2d {
                                ui.add(egui::DragValue::new(&mut faucet.direction.z).speed(0.05).prefix("Vz:"));
                            } else {
                                faucet.direction.z = 0.0;
                            }
                        });
                        match &mut faucet.shape {
                            FaucetShape::Circle { radius } => {
                                ui.add(
                                    egui::Slider::new(radius, 0.005..=0.40)
                                        .text("Radius (m)"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut faucet.spray_spread, 0.0..=0.5)
                                        .text("Spray Angle"),
                                );
                            }
                            FaucetShape::Rectangle { width, depth } => {
                                ui.add(
                                    egui::Slider::new(width, 0.05..=1.2).text("Width (m)"),
                                );
                                if !self.is_2d {
                                    ui.add(
                                        egui::Slider::new(depth, 0.01..=0.5).text("Depth (m)"),
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
                        ui.add(egui::Slider::new(&mut faucet.flow_rate, 25.0..=3000.0).text("Rate (p/s)"));
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
                });

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
                        ui.add(egui::Slider::new(&mut drain.radius, 0.04..=0.5).text("Radius"));
                    }
                }
                if let Some(idx) = to_remove {
                    self.solver.drains.remove(idx);
                }
            });
    }

    fn draw_physics_tab(&mut self, ui: &mut egui::Ui) {
        Self::section_card(ui, "Compute Acceleration", |ui| {
            if self.gpu_solver.is_some() {
                let prev = self.use_gpu;
                ui.checkbox(&mut self.use_gpu, "Enable Vulkan GPU Compute");
                if !prev && self.use_gpu {
                    if let Some(ref mut gpu) = self.gpu_solver {
                        gpu.sync_particles(
                            self.solver.bounds,
                            &self.solver.config,
                            &self.solver.particles,
                            true,
                        );
                    }
                }
                ui.weak(format!("Adapter: {}", self.gpu_device_name));
            } else {
                ui.weak("Vulkan GPU unavailable — using multi-core Rayon CPU solver.");
            }
        });

        ui.add_space(6.0);

        Self::section_card(ui, "Natural Fluid Dynamics (PBF)", |ui| {
            ui.add(
                egui::Slider::new(&mut self.solver.config.vorticity, 0.0..=0.18)
                    .text("Vorticity Swirl")
                    .fixed_decimals(3),
            )
            .on_hover_text("Macklin & Müller vorticity confinement: restores energetic eddies while keeping calm pools still");

            ui.add(
                egui::Slider::new(&mut self.solver.config.surface_tension, 0.0..=0.25)
                    .text("Surface Cohesion")
                    .fixed_decimals(3),
            )
            .on_hover_text("Akinci free-surface tension: forms rounded droplets, sheets, and cohesive splashes");

            ui.add(
                egui::Slider::new(&mut self.solver.config.viscosity, 0.0..=0.35)
                    .text("XSPH Viscosity")
                    .fixed_decimals(3),
            );

            ui.separator();

            ui.add(
                egui::Slider::new(&mut self.solver.config.gravity.y, -25.0..=5.0)
                    .text("Gravity Y (m/s²)"),
            );
            ui.add(
                egui::Slider::new(&mut self.solver.config.solver_iterations, 1..=8)
                    .text("Pressure Iterations"),
            );
            ui.add(
                egui::Slider::new(&mut self.solver.config.substeps, 1..=4).text("Substeps / Frame"),
            );
            ui.add(
                egui::Slider::new(&mut self.solver.config.boundary_friction, 0.5..=1.0)
                    .text("Wall Slip"),
            );
            ui.add(
                egui::Slider::new(&mut self.solver.config.restitution, 0.0..=0.2)
                    .text("Restitution"),
            );

            if ui.small_button("↺ Restore Natural Water Defaults").clicked() {
                let def = SolverConfig::default();
                self.solver.config.vorticity = def.vorticity;
                self.solver.config.surface_tension = def.surface_tension;
                self.solver.config.viscosity = def.viscosity;
                self.solver.config.gravity = def.gravity;
                self.solver.config.solver_iterations = def.solver_iterations;
                self.solver.config.substeps = def.substeps;
                self.solver.config.boundary_friction = def.boundary_friction;
                self.solver.config.restitution = def.restitution;
            }
        });

        ui.add_space(6.0);

        Self::section_card(ui, "Interactive Viewport Forces", |ui| {
            ui.weak(if self.is_2d {
                "Left Drag: Attract • Shift+Left Drag: Repel • Middle Drag: Pan"
            } else {
                "Left Drag: Attract • Shift+Left Drag: Repel • Middle: Pan • Right: Orbit"
            });
            ui.add(
                egui::Slider::new(&mut self.solver.mouse_force_strength, 10.0..=120.0)
                    .text("Cursor Force"),
            );
        });
    }

    fn draw_surface_tab(&mut self, ui: &mut egui::Ui) {
        // 1. Primary Render Mode Selector (Particles vs Surface Mesh)
        Self::section_card(ui, "Fluid Representation Mode (GUI & CLI)", |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_sized(
                        [142.0, 28.0],
                        egui::Button::new("⚪ SPH Spheres")
                            .selected(self.render_mode == RenderMode::Particles),
                    )
                    .clicked()
                {
                    self.render_mode = RenderMode::Particles;
                }
                if ui
                    .add_sized(
                        [142.0, 28.0],
                        egui::Button::new("💎 3D Surface Mesh")
                            .selected(self.render_mode == RenderMode::SurfaceMesh),
                    )
                    .clicked()
                {
                    self.render_mode = RenderMode::SurfaceMesh;
                    self.mesh_dirty = true;
                }
            });
            ui.weak(match self.render_mode {
                RenderMode::Particles => {
                    "Fast glossy instanced sphere impostors with Fresnel rim lighting."
                }
                RenderMode::SurfaceMesh => {
                    "Continuous Marching Cubes liquid isosurface with Beer-Lambert color, textures & PBR light."
                }
            });
        });

        ui.add_space(6.0);

        if self.render_mode == RenderMode::Particles {
            Self::section_card(ui, "Particle Sphere Shading & Colormap", |ui| {
                ui.checkbox(&mut self.lit_spheres, "Glossy Fresnel & Specular Spheres");
                ui.checkbox(&mut self.enable_culling, "Frustum Viewport Culling");
                ui.add(
                    egui::Slider::new(&mut self.display_size_factor, 0.5..=1.8)
                        .text("Sphere Scale"),
                );

                ui.separator();
                ui.label("Color Field:");
                egui::ComboBox::from_id_salt("p_color_mode")
                    .selected_text(self.color_mode.name())
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for &mode in ColorMode::ALL {
                            ui.selectable_value(&mut self.color_mode, mode, mode.name());
                        }
                    });

                ui.label("Colormap Gradient:");
                ui.horizontal(|ui| {
                    for &(cm, label) in &[
                        (ColorMapType::Ocean, "Ocean"),
                        (ColorMapType::Turbo, "Turbo"),
                        (ColorMapType::Viridis, "Viridis"),
                        (ColorMapType::Fire, "Fire"),
                    ] {
                        if ui.selectable_label(self.color_map == cm, label).clicked() {
                            self.color_map = cm;
                        }
                    }
                });
            });
            return;
        }

        // === Surface Mesh Controls (Material, Color, Texture, Light Behavior, Geometry) ===
        Self::section_card(ui, "Material Preset & Liquid Color", |ui| {
            let mut chosen_preset = self.mesh_config.preset;
            egui::ComboBox::from_id_salt("mesh_mat_preset")
                .selected_text(chosen_preset.name())
                .width(275.0)
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

            ui.add_space(4.0);
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
                .width(275.0)
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

        Self::section_card(ui, "Light Behavior & Optical Physics", |ui| {
            ui.horizontal(|ui| {
                ui.label("IOR Preset:");
                for &(ior_val, label) in &[
                    (1.333f32, "Water 1.33"),
                    (1.490f32, "Amber 1.49"),
                    (1.520f32, "Glass 1.52"),
                    (2.400f32, "Chrome 2.4"),
                ] {
                    if ui
                        .selectable_label((self.mesh_config.ior - ior_val).abs() < 0.02, label)
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
                egui::Slider::new(&mut self.mesh_config.subsurface_scattering, 0.0..=1.6)
                    .text("Subsurface Glow"),
            );
            ui.add(
                egui::Slider::new(&mut self.mesh_config.env_reflection, 0.0..=2.2)
                    .text("Sky Reflection"),
            );
            ui.add(
                egui::Slider::new(&mut self.mesh_config.metallic, 0.0..=1.0).text("Metallic"),
            );

            ui.separator();
            ui.label("Studio Key Sun & Shadows:");
            let mut sun_changed = false;
            sun_changed |= ui
                .add(
                    egui::Slider::new(&mut self.sun_azimuth_deg, -180.0..=180.0)
                        .text("Sun Azimuth (°)"),
                )
                .changed();
            sun_changed |= ui
                .add(
                    egui::Slider::new(&mut self.sun_elevation_deg, 8.0..=88.0)
                        .text("Sun Elevation (°)"),
                )
                .changed();
            if sun_changed {
                self.update_dir_from_sun_angles();
            }
            ui.add(
                egui::Slider::new(&mut self.mesh_config.light_intensity, 0.1..=3.0)
                    .text("Sun Radiance"),
            );
            ui.add(
                egui::Slider::new(&mut self.mesh_config.ambient_intensity, 0.05..=1.0)
                    .text("Ambient Fill"),
            );
            ui.checkbox(
                &mut self.mesh_config.cast_shadows,
                "Cast Colored Shadows & Caustics (CLI/MP4)",
            );
        });

        ui.add_space(6.0);

        Self::section_card(ui, "Isosurface Meshing Geometry", |ui| {
            let mut geom_changed = false;
            geom_changed |= ui
                .add(
                    egui::Slider::new(&mut self.mesh_config.voxel_scale, 0.45..=1.35)
                        .text("Voxel Scale"),
                )
                .on_hover_text("Smaller = finer surface triangles; larger = faster preview")
                .changed();
            geom_changed |= ui
                .add(
                    egui::Slider::new(&mut self.mesh_config.iso_value, 0.18..=0.75)
                        .text("Iso Threshold"),
                )
                .changed();
            geom_changed |= ui
                .add(
                    egui::Slider::new(&mut self.mesh_config.smoothing_passes, 0..=6)
                        .text("Taubin Smooth"),
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
                if ui.small_button("💾 Export .OBJ").clicked() {
                    self.export_current_surface_obj();
                }
            });
        });
    }

    fn draw_render_cli_tab(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // 1. Direct Studio MP4 Video Renderer
        Self::section_card(ui, "Studio MP4 Video Renderer (GUI)", |ui| {
            ui.horizontal(|ui| {
                ui.label("Render Mode:");
                ui.selectable_value(&mut self.render_mode, RenderMode::Particles, "⚪ Spheres");
                if ui
                    .selectable_value(
                        &mut self.render_mode,
                        RenderMode::SurfaceMesh,
                        "💎 Surface Mesh",
                    )
                    .clicked()
                {
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

            ui.add(
                egui::Slider::new(&mut self.cli_duration, 0.5..=30.0)
                    .text("Duration (s)")
                    .fixed_decimals(1),
            );

            ui.horizontal(|ui| {
                ui.label("FPS:");
                for &f in &[30u32, 60, 120] {
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
                if ui.button("⏹ Stop & Save Current Progress").clicked() {
                    self.bg_render_cancel.store(true, Ordering::SeqCst);
                }
                ctx.request_repaint();
            } else {
                let render_btn = egui::Button::new(
                    egui::RichText::new("🎬 Render MP4 Video Now")
                        .strong()
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(20, 118, 212));

                if ui.add_sized([ui.available_width(), 30.0], render_btn).clicked() {
                    self.start_gui_mp4_render();
                }
            }
        });

        ui.add_space(6.0);

        // 2. CLI Command Builder
        Self::section_card(ui, "CLI Command Builder", |ui| {
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
                if ui.button("📋 Copy CLI Command").clicked() {
                    ctx.copy_text(cmd.clone());
                    self.status_message = "Copied CLI command to clipboard!".to_string();
                }
                if ui.button("💎 Copy Mesh MP4 CLI").clicked() {
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

        ui.add_space(6.0);

        // 3. Blender PLY & Wavefront OBJ Export
        Self::section_card(ui, "Geometry & Point Cloud Export", |ui| {
            ui.horizontal(|ui| {
                ui.label("Folder:");
                ui.text_edit_singleline(&mut self.export_path);
            });
            ui.horizontal(|ui| {
                if ui.button("💾 Save Frame (.ply)").clicked() {
                    self.export_current_frame();
                }
                if ui.button("💎 Save Surface (.obj)").clicked() {
                    self.export_current_surface_obj();
                }
            });
            ui.checkbox(&mut self.export_sequence, "Auto-record PLY sequence");
            if self.export_sequence {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 185, 65),
                    format!("● Recording frame {:05}", self.exported_frames_count),
                );
            }
        });
    }
}

impl eframe::App for LiquidApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            Self::apply_studio_theme(ctx);
            self.theme_applied = true;
        }

        // Keyboard shortcuts (Space to toggle play/pause when not typing in text fields)
        if !ctx.wants_keyboard_input() && ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.is_playing = !self.is_playing;
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
                    ui.label(
                        egui::RichText::new("🌊 LiquidRust")
                            .strong()
                            .size(15.5)
                            .color(egui::Color32::from_rgb(235, 244, 255)),
                    );

                    // Version pill / Update badge right next to app title
                    let upd_snap = self.updater.snapshot();
                    match &upd_snap.phase {
                        UpdatePhase::UpdateAvailable(info) => {
                            let upd_btn = egui::Button::new(
                                egui::RichText::new(format!("⬆ Update {}", info.tag_name))
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
                                egui::RichText::new(format!("⬇ {}%", pct))
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

                    let active_fill = egui::Color32::from_rgb(14, 116, 184);
                    let idle_fill = egui::Color32::from_rgb(32, 41, 54);
                    let btn_3d = egui::Button::new(
                        egui::RichText::new("🌐 3D")
                            .strong()
                            .size(13.0)
                            .color(if !self.is_2d { egui::Color32::WHITE } else { egui::Color32::from_rgb(175, 190, 210) }),
                    )
                    .fill(if !self.is_2d { active_fill } else { idle_fill });
                    if ui
                        .add(btn_3d)
                        .on_hover_text("Switch to 3D Volumetric simulation")
                        .clicked()
                        && self.is_2d
                    {
                        self.set_2d(false);
                    }

                    let btn_2d = egui::Button::new(
                        egui::RichText::new("▭ 2D")
                            .strong()
                            .size(13.0)
                            .color(if self.is_2d { egui::Color32::WHITE } else { egui::Color32::from_rgb(175, 190, 210) }),
                    )
                    .fill(if self.is_2d { active_fill } else { idle_fill });
                    if ui
                        .add(btn_2d)
                        .on_hover_text("Switch to 2D Planar simulation")
                        .clicked()
                        && !self.is_2d
                    {
                        self.set_2d(true);
                    }

                    ui.separator();

                    let play_label = if self.is_playing {
                        "⏸ Pause"
                    } else {
                        "▶ Play"
                    };
                    if ui
                        .button(egui::RichText::new(play_label).strong())
                        .on_hover_text("Play / Pause simulation (Space)")
                        .clicked()
                    {
                        self.is_playing = !self.is_playing;
                    }

                    if ui
                        .button("⏭ Step")
                        .on_hover_text("Advance single simulation frame")
                        .clicked()
                    {
                        let dt = self.fixed_dt * self.sim_speed;
                        self.step_simulation_once(dt);
                        ctx.request_repaint();
                    }

                    if ui.button("↺ Reset").clicked() {
                        self.load_preset(self.current_preset);
                    }

                    if ui
                        .button("🎥 Cam")
                        .on_hover_text("Reset camera framing")
                        .clicked()
                    {
                        if self.is_2d {
                            self.camera.azimuth = 0.0;
                            self.camera.elevation = 0.0;
                            self.camera.target = Vec3::new(0.0, 0.6, 0.0);
                            self.camera.distance = 2.4;
                        } else {
                            self.camera.reset();
                        }
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
                    if ui
                        .selectable_label(
                            self.render_mode == RenderMode::Particles,
                            "⚪ Spheres",
                        )
                        .clicked()
                    {
                        self.render_mode = RenderMode::Particles;
                    }
                    if ui
                        .selectable_label(
                            self.render_mode == RenderMode::SurfaceMesh,
                            "💎 Mesh",
                        )
                        .clicked()
                    {
                        self.render_mode = RenderMode::SurfaceMesh;
                        self.mesh_dirty = true;
                        self.active_tab = InspectorTab::Surface;
                    }

                    ui.separator();

                    // Compact Speed pills
                    for &(spd, lbl) in &[(0.25f32, "0.25x"), (0.5f32, "0.5x"), (1.0f32, "1.0x")] {
                        if ui
                            .selectable_label((self.sim_speed - spd).abs() < 0.03, lbl)
                            .clicked()
                        {
                            self.sim_speed = spd;
                        }
                    }

                    // Right-aligned zero-jitter monospace telemetry pills
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button("📋 CLI")
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
                    let backend_str = if self.use_gpu && self.gpu_solver.is_some() {
                        format!("⚡ Vulkan: {}", self.gpu_device_name)
                    } else {
                        "⚙ CPU (Rayon Multi-Core)".to_string()
                    };
                    ui.label(
                        egui::RichText::new(backend_str)
                            .size(11.0)
                            .color(egui::Color32::from_rgb(112, 184, 255)),
                    );
                    ui.separator();
                    ui.label(
                        egui::RichText::new(&self.status_message)
                            .size(11.0)
                            .color(egui::Color32::from_rgb(190, 202, 222)),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(if self.is_2d {
                            "Space: Play/Pause • Left-Drag: Attract • Shift+Left-Drag: Repel • Middle-Drag: Pan • Scroll: Zoom"
                        } else {
                            "Space: Play/Pause • Left-Drag: Attract • Shift+Left-Drag: Repel • Middle-Drag: Pan • Right-Drag: Orbit"
                        });
                    });
                });
            });

        // 3. Left Studio Inspector Panel with 4 Organized Tabs
        egui::SidePanel::left("studio_inspector")
            .default_width(326.0)
            .min_width(305.0)
            .max_width(420.0)
            .show(ctx, |ui| {
                ui.add_space(3.0);
                ui.horizontal(|ui| {
                    for &(tab, label) in &[
                        (InspectorTab::Scene, "🌊 Scene"),
                        (InspectorTab::Physics, "⚙ Physics"),
                        (InspectorTab::Surface, "💎 Surface"),
                        (InspectorTab::RenderCli, "🎬 Render"),
                    ] {
                        if ui
                            .add_sized(
                                [74.0, 26.0],
                                egui::Button::new(label).selected(self.active_tab == tab),
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
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(9, 13, 20)))
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
                    // Populate persistent vertex buffer with parallel frustum culling (no redundant CPU sort!)
                    let enable_culling = self.enable_culling;
                    let color_mode = self.color_mode;
                    let color_map = self.color_map;
                    let rest_density = self.solver.config.rest_density;
                    let sim_bounds = self.solver.bounds;
                    self.render_vertices.clear();
                    self.render_vertices.par_extend(
                        self.solver.particles.par_iter().filter_map(|p| {
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
                            let t = match color_mode {
                                ColorMode::Velocity => (p.speed() / 2.4).clamp(0.0, 1.0),
                                ColorMode::Pressure => {
                                    (p.pressure / (rest_density * 0.4)).clamp(0.0, 1.0)
                                }
                                ColorMode::Density => {
                                    let dev = (p.density - rest_density).abs() / rest_density;
                                    (dev * 4.0).clamp(0.0, 1.0)
                                }
                                ColorMode::Height => ((p.position.y - sim_bounds.min.y)
                                    / (sim_bounds.max.y - sim_bounds.min.y))
                                    .clamp(0.0, 1.0),
                                ColorMode::KineticEnergy => {
                                    (p.kinetic_energy() / 4.0).clamp(0.0, 1.0)
                                }
                                ColorMode::SingleColor => -1.0,
                            };
                            let c = if t < 0.0 {
                                [30, 140, 225]
                            } else {
                                ColorMap::sample(color_map, t)
                            };
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
            egui::Window::new("⚠️ Warning: Extreme Simulation Quality")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_max_width(450.0);
                    ui.add_space(4.0);
                    ui.vertical_centered(|ui| {
                        ui.heading("⚠️ 1,000,000+ Particles Workload");
                    });
                    ui.add_space(8.0);
                    ui.label(
                        "You are about to switch to the Extreme Quality preset, which spawns over \
                        1,000,000 physical fluid particles in a dense microscopic lattice.",
                    );
                    ui.add_space(6.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 120, 40),
                        "⚡ System Performance & Freeze Warning:",
                    );
                    ui.label(
                        "• Computing over 1M particles places a massive load on your GPU and system RAM.\n\
                        • Your computer or desktop window compositor may temporarily freeze, stutter, or lag.\n\
                        • If using CPU fallback, this could take minutes per frame. Vulkan GPU compute is strongly recommended.",
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
                                egui::RichText::new("⚠️ Proceed to 1M+")
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

        // 6. GitHub Release Self-Updater Notification & Progress Modal
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
                format!("⬆ Update Available — {}", info.tag_name)
            }
            UpdatePhase::Downloading { info, .. } => {
                format!("⬇ Downloading Update {}...", info.tag_name)
            }
            UpdatePhase::Installing(info) => {
                format!("⚙ Installing {}...", info.tag_name)
            }
            _ => "🔄 LiquidRust Software Update".to_string(),
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
                                "✔ You are running the latest version (v{}).",
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
                            if ui.button("🔄 Check Again").clicked() {
                                self.updater.spawn_check(Some(ctx.clone()), true);
                            }
                            if ui
                                .button("↻ Reinstall & Restart")
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
                                egui::RichText::new(format!("Installed: v{}", CURRENT_VERSION))
                                    .monospace()
                                    .color(egui::Color32::from_rgb(168, 184, 206)),
                            );
                            ui.label("→");
                            ui.label(
                                egui::RichText::new(format!("Latest: {}", info.tag_name))
                                    .monospace()
                                    .strong()
                                    .color(egui::Color32::from_rgb(74, 222, 128)),
                            );
                        });
                        ui.weak(format!(
                            "{} • {} ({})",
                            info.title,
                            info.asset_name,
                            format_bytes(info.asset_size)
                        ));

                        if !info.notes.is_empty() {
                            ui.add_space(8.0);
                            egui::Frame::group(ui.style())
                                .fill(egui::Color32::from_rgb(10, 14, 22))
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    egui::ScrollArea::vertical()
                                        .max_height(145.0)
                                        .show(ui, |ui| {
                                            ui.label(
                                                egui::RichText::new(&info.notes)
                                                    .size(12.0)
                                                    .color(egui::Color32::from_rgb(206, 218, 235)),
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
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let update_btn = egui::Button::new(
                                    egui::RichText::new("⬆ Update & Restart")
                                        .strong()
                                        .color(egui::Color32::from_rgb(8, 24, 16)),
                                )
                                .fill(egui::Color32::from_rgb(74, 222, 128));
                                if ui.add(update_btn).clicked() {
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
                        ui.label(
                            egui::RichText::new(format!(
                                "Downloading {} ({})...",
                                info.asset_name, info.tag_name
                            ))
                            .strong(),
                        );
                        ui.add_space(6.0);
                        let frac = if *total_bytes > 0 {
                            (*downloaded_bytes as f32 / *total_bytes as f32).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .show_percentage()
                                .animate(true),
                        );
                        ui.add_space(4.0);
                        ui.weak(format!(
                            "{} / {} — LiquidRust will restart automatically when ready.",
                            format_bytes(*downloaded_bytes),
                            format_bytes(*total_bytes)
                        ));
                    }
                    UpdatePhase::Installing(info) => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(
                                egui::RichText::new(format!(
                                    "Applying update {} & restarting...",
                                    info.tag_name
                                ))
                                .strong(),
                            );
                        });
                    }
                    UpdatePhase::Error(err) => {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 110, 110),
                            egui::RichText::new("Update Error").strong(),
                        );
                        ui.label(err);
                        ui.add_space(10.0);
                        ui.separator();
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("🔄 Retry").clicked() {
                                self.updater.spawn_check(Some(ctx.clone()), true);
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button("Close").clicked() {
                                    self.updater.close_modal();
                                }
                            });
                        });
                    }
                    UpdatePhase::ReadyToRestart { .. } => {}
                }
            });
    }
}
