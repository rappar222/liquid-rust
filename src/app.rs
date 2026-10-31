use crate::export::PlyExporter;
use crate::gpu::GpuSphSolver;
use crate::math::{unproject_mouse_ray, Aabb};
use crate::render::{Camera, ColorMap, ColorMapType, ColorMode, GlRenderer, ParticleVertex};
use crate::sph::collider::Collider;
use crate::sph::scene::{SceneGenerator, ScenePreset};
use crate::sph::solver::{compute_lattice_rest_density, SimulationQuality, SolverConfig, SphSolver};
use eframe::egui;
use glam::Vec3;
use glow::HasContext as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

    // Rendering settings
    pub renderer: Arc<Mutex<Option<GlRenderer>>>,
    pub lit_spheres: bool,
    pub enable_culling: bool,
    pub drawn_particles_count: usize,
    pub color_mode: ColorMode,
    pub color_map: ColorMapType,
    pub display_size_factor: f32,
    pub sphere_obstacle_pos: Vec3,
    pub sphere_obstacle_radius: f32,
    pub sphere_obstacle_enabled: bool,

    // Diagnostics
    pub last_frame_time: Instant,
    pub frame_counter: usize,
    pub current_fps: f32,
    pub last_step_time_ms: f32,

    // Export
    pub export_path: String,
    pub export_sequence: bool,
    pub exported_frames_count: usize,
    pub status_message: String,
    pub show_extreme_modal: bool,
}

impl LiquidApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
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

        let (gpu_solver, gpu_device_name, use_gpu) = match GpuSphSolver::new(bounds, &solver.config, &solver.particles) {
            Ok(g) => {
                let name = g.device_name.clone();
                (Some(g), name, true)
            }
            Err(e) => {
                eprintln!("GPU solver init for GUI: {}", e);
                (None, "GPU unavailable".to_string(), false)
            }
        };

        Self {
            solver,
            gpu_solver,
            use_gpu,
            gpu_device_name,
            camera: Camera::default(),
            is_playing: true,
            fixed_dt: 0.016,
            sim_speed: 1.0,
            current_preset: preset,
            current_quality: SimulationQuality::Medium,
            renderer,
            lit_spheres: true,
            enable_culling: true,
            drawn_particles_count: 0,
            color_mode: ColorMode::Velocity,
            color_map: ColorMapType::Ocean,
            display_size_factor: 1.0,
            sphere_obstacle_pos: Vec3::new(0.0, 0.45, 0.0),
            sphere_obstacle_radius: 0.22,
            sphere_obstacle_enabled: false,
            last_frame_time: Instant::now(),
            frame_counter: 0,
            current_fps: 60.0,
            last_step_time_ms: 0.0,
            export_path: "exports".to_string(),
            export_sequence: false,
            exported_frames_count: 0,
            status_message: "Ready".to_string(),
            show_extreme_modal: false,
        }
    }

    pub fn load_preset(&mut self, preset: ScenePreset) {
        self.current_preset = preset;
        SceneGenerator::setup_scene(&mut self.solver, preset);
        if preset == ScenePreset::ObstacleDrop {
            self.sphere_obstacle_enabled = true;
        }
        if let Some(ref mut gpu) = self.gpu_solver {
            gpu.init_buffers(self.solver.bounds, &self.solver.config, &self.solver.particles);
        }
        self.status_message = format!("Loaded preset: {}", preset.name());
    }

    pub fn set_quality(&mut self, quality: SimulationQuality) {
        self.current_quality = quality;
        self.solver.apply_quality(quality);
        self.load_preset(self.current_preset);
        self.status_message = format!("Set quality: {}", quality.name());
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
                self.status_message = format!("Exported frame to {}", path.display());
                self.exported_frames_count += 1;
            }
            Err(e) => {
                self.status_message = format!("Export error: {}", e);
            }
        }
    }

    fn compute_particle_color(&self, p: &crate::sph::particle::Particle) -> [u8; 3] {
        let t = match self.color_mode {
            ColorMode::Velocity => {
                // Calm resting water: 0..0.4 m/s -> deep oceanic blue
                // Moving wave: 0.5..1.8 m/s -> azure/turquoise
                // Splash / impact: > 2.0 m/s -> white foam
                (p.speed() / 2.4).clamp(0.0, 1.0)
            }
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
}

impl eframe::App for LiquidApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Calculate FPS
        self.frame_counter += 1;
        let elapsed = self.last_frame_time.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            self.current_fps = self.frame_counter as f32 / elapsed;
            self.frame_counter = 0;
            self.last_frame_time = Instant::now();
        }

        // Simulation update step
        if self.is_playing {
            let start = Instant::now();
            let effective_dt = self.fixed_dt * self.sim_speed;
            if self.use_gpu && self.gpu_solver.is_some() {
                if let Some(ref mut gpu) = self.gpu_solver {
                    gpu.step(
                        self.solver.bounds,
                        &self.solver.config,
                        &self.solver.colliders,
                        self.solver.mouse_ray,
                        self.solver.mouse_force_strength,
                        self.solver.mouse_force_mode,
                        effective_dt,
                    );
                    gpu.read_particles(&mut self.solver.particles);
                }
            } else {
                self.solver.step(effective_dt);
            }
            self.last_step_time_ms = start.elapsed().as_secs_f32() * 1000.0;

            if self.export_sequence {
                self.export_current_frame();
            }

            ctx.request_repaint();
        }

        // 1. Top Panel: Toolbar
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🌊 LiquidRust 3D");
                ui.separator();

                if ui
                    .button(if self.is_playing { "⏸ Pause" } else { "▶ Play" })
                    .clicked()
                {
                    self.is_playing = !self.is_playing;
                }

                if ui.button("⏭ Step").clicked() {
                    let start = Instant::now();
                    let effective_dt = self.fixed_dt * self.sim_speed;
                    if self.use_gpu && self.gpu_solver.is_some() {
                        if let Some(ref mut gpu) = self.gpu_solver {
                            gpu.step(
                                self.solver.bounds,
                                &self.solver.config,
                                &self.solver.colliders,
                                self.solver.mouse_ray,
                                self.solver.mouse_force_strength,
                                self.solver.mouse_force_mode,
                                effective_dt,
                            );
                            gpu.read_particles(&mut self.solver.particles);
                        }
                    } else {
                        self.solver.step(effective_dt);
                    }
                    self.last_step_time_ms = start.elapsed().as_secs_f32() * 1000.0;
                    ctx.request_repaint();
                }

                if ui.button("🔄 Reset").clicked() {
                    self.load_preset(self.current_preset);
                }

                ui.separator();

                // ASMR Slow Motion / Speed Controls
                ui.label("⏱ Speed:");
                ui.add(
                    egui::Slider::new(&mut self.sim_speed, 0.05..=2.0)
                        .show_value(true)
                        .fixed_decimals(2),
                );
                if ui
                    .selectable_label((self.sim_speed - 0.25).abs() < 0.02, "🎬 0.25x ASMR")
                    .clicked()
                {
                    self.sim_speed = 0.25;
                }
                if ui
                    .selectable_label((self.sim_speed - 0.50).abs() < 0.02, "0.5x")
                    .clicked()
                {
                    self.sim_speed = 0.50;
                }
                if ui
                    .selectable_label((self.sim_speed - 1.00).abs() < 0.02, "1.0x")
                    .clicked()
                {
                    self.sim_speed = 1.00;
                }

                ui.separator();

                // Quick preset selector
                egui::ComboBox::from_id_salt("preset_combo")
                    .selected_text(self.current_preset.name())
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
                if self.enable_culling {
                    ui.label(format!("Particles: {} (Drawn: {})", self.solver.particle_count(), self.drawn_particles_count));
                } else {
                    ui.label(format!("Particles: {}", self.solver.particle_count()));
                }
                ui.separator();
                ui.label(format!("FPS: {:.0}", self.current_fps));
                ui.separator();
                ui.label(format!("Sim: {:.1} ms", self.last_step_time_ms));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("📷 Export PLY").clicked() {
                        self.export_current_frame();
                    }
                });
            });
        });

        // 2. Left Panel: Controls and Parameters
        egui::SidePanel::left("left_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(4.0);

                    // --- Compute Backend ---
                    ui.collapsing("⚡ Compute Acceleration", |ui| {
                        if self.gpu_solver.is_some() {
                            ui.checkbox(&mut self.use_gpu, "Enable GPU Acceleration");
                            ui.label(format!("Device: {}", self.gpu_device_name));
                            ui.label("Backend: Vulkan Compute");
                        } else {
                            ui.label("GPU Compute: Unavailable");
                            ui.label("Using: Multi-threaded CPU");
                        }
                    });

                    ui.separator();

                    // --- Presets ---
                    ui.collapsing("📁 Scene Presets", |ui| {
                        for &preset in ScenePreset::ALL {
                            let is_current = self.current_preset == preset;
                            if ui.selectable_label(is_current, preset.name()).clicked() {
                                self.load_preset(preset);
                            }
                        }
                    });

                    ui.separator();

                    // --- Particle Quality / Resolution ---
                    ui.collapsing("💎 Particle Quality & Smoothness", |ui| {
                        ui.label("Quality Presets (Particle Size):");
                        for &q in SimulationQuality::ALL {
                            let is_current = self.current_quality == q;
                            let btn = if q == SimulationQuality::Extreme {
                                ui.selectable_label(
                                    is_current,
                                    egui::RichText::new(q.name())
                                        .color(egui::Color32::from_rgb(255, 120, 40)),
                                )
                            } else {
                                ui.selectable_label(is_current, q.name())
                            };
                            if btn.clicked() {
                                if q == SimulationQuality::Extreme && !is_current {
                                    self.show_extreme_modal = true;
                                } else {
                                    self.set_quality(q);
                                }
                            }
                        }

                        ui.separator();
                        ui.label("Fine Granularity:");
                        let mut changed = false;
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.solver.config.particle_radius, 0.010..=0.040)
                                    .text("Radius (R)"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut self.solver.config.spacing, 0.018..=0.065)
                                    .text("Spacing (d)"),
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

                        if ui.button("⚡ Respawn Particles with Current Size").clicked() {
                            let h = self.solver.config.smoothing_length;
                            self.solver.set_smoothing_length(h);
                            self.load_preset(self.current_preset);
                        }
                    });

                    ui.separator();

                    // --- Physics Parameters ---
                    ui.collapsing("⚙ Fluid Physics (PBF)", |ui| {
                        ui.add(
                            egui::Slider::new(&mut self.sim_speed, 0.05..=2.0)
                                .text("Speed / ASMR Slow-Mo")
                                .fixed_decimals(2),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.viscosity, 0.0..=0.5)
                                .text("Viscosity (Damping)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.surface_tension, 0.0..=0.2)
                                .text("Surface Tension"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.restitution, 0.0..=0.2)
                                .text("Restitution (Bounce)"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.solver_iterations, 1..=6)
                                .text("Solver Iterations"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.substeps, 1..=4)
                                .text("Substeps"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.gravity.y, -25.0..=5.0)
                                .text("Gravity Y"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.solver.config.boundary_friction,
                                0.5..=1.0,
                            )
                            .text("Wall Friction"),
                        );
                    });

                    ui.separator();

                    // --- Obstacle Controls ---
                    ui.collapsing("🛑 Solid Obstacle", |ui| {
                        let prev_enabled = self.sphere_obstacle_enabled;
                        ui.checkbox(&mut self.sphere_obstacle_enabled, "Sphere Obstacle");

                        let mut obstacle_changed = false;
                        if self.sphere_obstacle_enabled {
                            obstacle_changed |= ui
                                .add(
                                    egui::Slider::new(&mut self.sphere_obstacle_pos.x, -0.6..=0.6)
                                        .text("Pos X"),
                                )
                                .changed();
                            obstacle_changed |= ui
                                .add(
                                    egui::Slider::new(&mut self.sphere_obstacle_pos.y, 0.05..=1.0)
                                        .text("Pos Y"),
                                )
                                .changed();
                            obstacle_changed |= ui
                                .add(
                                    egui::Slider::new(&mut self.sphere_obstacle_pos.z, -0.4..=0.4)
                                        .text("Pos Z"),
                                )
                                .changed();
                            obstacle_changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.sphere_obstacle_radius,
                                        0.08..=0.4,
                                    )
                                    .text("Radius"),
                                )
                                .changed();
                        }

                        if prev_enabled != self.sphere_obstacle_enabled || obstacle_changed {
                            self.solver.colliders.clear();
                            if self.sphere_obstacle_enabled {
                                self.solver.colliders.push(Collider::sphere(
                                    self.sphere_obstacle_pos,
                                    self.sphere_obstacle_radius,
                                ));
                            }
                        }
                    });

                    ui.separator();

                    // --- Visuals & Color ---
                    ui.collapsing("🎨 Rendering & Colors", |ui| {
                        ui.checkbox(&mut self.lit_spheres, "3D Shaded Spheres");
                        ui.checkbox(&mut self.enable_culling, "View Frustum Culling");
                        ui.add(
                            egui::Slider::new(&mut self.display_size_factor, 0.5..=1.8)
                                .text("Particle Size"),
                        );

                        ui.label("Color Mode:");
                        for &mode in ColorMode::ALL {
                            if ui
                                .selectable_label(self.color_mode == mode, mode.name())
                                .clicked()
                            {
                                self.color_mode = mode;
                            }
                        }

                        ui.separator();
                        ui.label("Color Map:");
                        ui.horizontal(|ui| {
                            if ui
                                .selectable_label(self.color_map == ColorMapType::Ocean, "Ocean")
                                .clicked()
                            {
                                self.color_map = ColorMapType::Ocean;
                            }
                            if ui
                                .selectable_label(self.color_map == ColorMapType::Turbo, "Turbo")
                                .clicked()
                            {
                                self.color_map = ColorMapType::Turbo;
                            }
                            if ui
                                .selectable_label(self.color_map == ColorMapType::Viridis, "Viridis")
                                .clicked()
                            {
                                self.color_map = ColorMapType::Viridis;
                            }
                            if ui
                                .selectable_label(self.color_map == ColorMapType::Fire, "Fire")
                                .clicked()
                            {
                                self.color_map = ColorMapType::Fire;
                            }
                        });
                    });

                    ui.separator();

                    // --- Interactive Mouse Tool ---
                    ui.collapsing("🖐 Mouse Interaction", |ui| {
                        ui.label("Hold Shift + Drag in 3D Viewport:");
                        ui.label("• Left-Click Drag: Attract fluid");
                        ui.label("• Right-Click Drag: Repel fluid");
                        ui.add(
                            egui::Slider::new(&mut self.solver.mouse_force_strength, 10.0..=120.0)
                                .text("Interaction Force"),
                        );
                        if ui.button("Reset Camera View").clicked() {
                            self.camera.reset();
                        }
                    });

                    ui.separator();

                    // --- PLY Export ---
                    ui.collapsing("💾 Blender / PLY Exporter", |ui| {
                        ui.label("Export directory:");
                        ui.text_edit_singleline(&mut self.export_path);
                        if ui.button("Save Current Frame (.ply)").clicked() {
                            self.export_current_frame();
                        }
                        ui.checkbox(&mut self.export_sequence, "Record PLY sequence");
                        if self.export_sequence {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 180, 50),
                                format!("Recording frame {:05}...", self.exported_frames_count),
                            );
                        }
                    });

                    ui.add_space(8.0);
                    ui.colored_label(
                        egui::Color32::LIGHT_GRAY,
                        format!("Status: {}", self.status_message),
                    );
                });
            });

        // 3. Central Viewport: Custom OpenGL 3D Render
        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = ui.available_size();
            let (rect, response) =
                ui.allocate_exact_size(available_size, egui::Sense::click_and_drag());

            let is_shift_held = ctx.input(|i| i.modifiers.shift);

            // Handle Camera Navigation
            if !is_shift_held {
                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    self.camera.rotate(delta.x * 0.006, -delta.y * 0.006);
                } else if response.dragged_by(egui::PointerButton::Secondary)
                    || response.dragged_by(egui::PointerButton::Middle)
                {
                    let delta = response.drag_delta();
                    self.camera.pan(delta.x, delta.y);
                }
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

            // Handle Fluid Interaction with Shift + Mouse
            if is_shift_held && response.hovered() {
                if let Some(hover_pos) = response.hover_pos() {
                    let local_x = hover_pos.x - rect.min.x;
                    let local_y = hover_pos.y - rect.min.y;
                    let ray = unproject_mouse_ray(
                        glam::vec2(local_x, local_y),
                        glam::vec2(rect.width(), rect.height()),
                        view,
                        proj,
                    );
                    self.solver.mouse_ray = Some(ray);

                    if response.dragged_by(egui::PointerButton::Primary) {
                        self.solver.mouse_force_mode = 1; // attract
                    } else if response.dragged_by(egui::PointerButton::Secondary) {
                        self.solver.mouse_force_mode = -1; // repel
                    } else {
                        self.solver.mouse_force_mode = 0;
                    }
                }
            } else {
                self.solver.mouse_force_mode = 0;
            }

            // Prepare vertex data with optional view frustum culling
            let enable_culling = self.enable_culling;
            let particles_data: Vec<ParticleVertex> = self
                .solver
                .particles
                .iter()
                .filter(|p| {
                    if !enable_culling {
                        return true;
                    }
                    let p_world = glam::Vec4::new(p.position.x, p.position.y, p.position.z, 1.0);
                    let clip = view_proj * p_world;
                    if clip.w <= 0.001 {
                        return false;
                    }
                    let ndc_x = clip.x / clip.w;
                    let ndc_y = clip.y / clip.w;
                    let ndc_z = clip.z / clip.w;
                    let margin = 0.2;
                    ndc_x >= -1.0 - margin
                        && ndc_x <= 1.0 + margin
                        && ndc_y >= -1.0 - margin
                        && ndc_y <= 1.0 + margin
                        && ndc_z >= -1.0
                        && ndc_z <= 1.0
                })
                .map(|p| {
                    let c = self.compute_particle_color(p);
                    ParticleVertex {
                        pos: p.position.to_array(),
                        color: [
                            c[0] as f32 / 255.0,
                            c[1] as f32 / 255.0,
                            c[2] as f32 / 255.0,
                            1.0,
                        ],
                    }
                })
                .collect();

            self.drawn_particles_count = particles_data.len();

            let renderer_arc = self.renderer.clone();
            let bounds = self.solver.bounds;
            let colliders = self.solver.colliders.clone();
            let lit_spheres = self.lit_spheres;
            let particle_radius =
                self.solver.config.particle_radius * self.display_size_factor;

            let callback = egui_glow::CallbackFn::new(move |info, painter| {
                if let Ok(mut guard) = renderer_arc.lock() {
                    if let Some(ref mut renderer) = *guard {
                        let gl = painter.gl();
                        let ppi = info.pixels_per_point;
                        let vp_x = (rect.min.x * ppi) as i32;
                        let vp_y = (info.screen_size_px[1] as f32 - rect.max.y * ppi) as i32;
                        let vp_w = (rect.width() * ppi) as i32;
                        let vp_h = (rect.height() * ppi) as i32;

                        unsafe {
                            gl.viewport(vp_x, vp_y, vp_w, vp_h);
                        }

                        renderer.render(
                            gl,
                            view_proj,
                            [rect.width() * ppi, rect.height() * ppi],
                            particle_radius,
                            fov_y,
                            &particles_data,
                            lit_spheres,
                            bounds,
                            &colliders,
                        );
                    }
                }
            });

            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(callback),
            });
        });

        // 4. Extreme Quality Confirmation Modal Pop-up
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
                        1,000,000 physical fluid particles in a dense microscopic lattice."
                    );
                    ui.add_space(6.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 120, 40),
                        "⚡ System Performance & Freeze Warning:",
                    );
                    ui.label(
                        "• Computing over 1M particles places a massive load on your GPU and system RAM.\n\
                        • Your computer or desktop window compositor may temporarily freeze, stutter, or lag.\n\
                        • If using CPU fallback, this could take minutes per frame. Vulkan GPU compute is strongly recommended."
                    );
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("❌ Cancel (Keep Current Quality)").clicked() {
                            self.show_extreme_modal = false;
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let proceed_btn = egui::Button::new(
                                egui::RichText::new("⚠️ I Agree, Proceed to 1M+")
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
    }
}
