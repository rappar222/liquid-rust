use crate::export::PlyExporter;
use crate::gpu::GpuSphSolver;
use crate::math::{unproject_mouse_ray, Aabb};
use crate::render::{Camera, ColorMap, ColorMapType, ColorMode, GlRenderer, ParticleVertex};
use crate::sph::{ChamberShape, Collider, ColliderShape, CubeFace, Drain, Faucet, FaucetShape};
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
    pub obj_file_path: String,
    pub next_collider_id: u32,
    pub next_faucet_id: u32,
    pub next_drain_id: u32,
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
            obj_file_path: "model.obj".to_string(),
            next_collider_id: 1,
            next_faucet_id: 1,
            next_drain_id: 1,
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
                self.solver.emit_faucets(effective_dt);
                if let Some(ref mut gpu) = self.gpu_solver {
                    if gpu.particle_count != self.solver.particles.len() || self.solver.particles.len() > gpu.particle_capacity {
                        gpu.init_buffers(self.solver.bounds, &self.solver.config, &self.solver.particles);
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
                    );
                    gpu.read_particles(&mut self.solver.particles);
                }

                // Resolve any Mesh (.obj) colliders on readback/particles
                let radius = self.solver.config.particle_radius;
                for col in &self.solver.colliders {
                    if let ColliderShape::Mesh(_) = &col.shape {
                        for p in &mut self.solver.particles {
                            col.resolve(&mut p.position, &mut p.velocity, radius);
                        }
                    }
                }

                let drained = self.solver.drain_particles();
                if drained > 0 {
                    if let Some(ref mut gpu) = self.gpu_solver {
                        gpu.init_buffers(self.solver.bounds, &self.solver.config, &self.solver.particles);
                    }
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
                                &self.solver.chamber,
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
                    egui::Slider::new(&mut self.sim_speed, 0.01..=2.0)
                        .show_value(true)
                        .fixed_decimals(3),
                );
                if ui
                    .selectable_label((self.sim_speed - 0.0625).abs() < 0.005, "🐌 16x Slow")
                    .clicked()
                {
                    self.sim_speed = 0.0625;
                }
                if ui
                    .selectable_label((self.sim_speed - 0.25).abs() < 0.02, "🎬 4x Slow")
                    .clicked()
                {
                    self.sim_speed = 0.25;
                }
                if ui
                    .selectable_label((self.sim_speed - 0.50).abs() < 0.02, "2x Slow")
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

                    // --- Simulation Chamber & Shape Controls ---
                    ui.collapsing("🛁 Simulation Chamber & Bounds", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Shape:");
                            for &s in &ChamberShape::ALL {
                                if ui.selectable_label(self.solver.chamber.shape == s, s.name()).clicked() {
                                    self.solver.chamber.shape = s;
                                    self.solver.bounds = self.solver.chamber.compute_bounds();
                                    if let Some(ref mut gpu) = self.gpu_solver {
                                        gpu.init_buffers(self.solver.bounds, &self.solver.config, &self.solver.particles);
                                    }
                                }
                            }
                        });

                        ui.separator();
                        ui.label("Chamber Scale:");
                        let mut scale_changed = false;

                        // Uniform scale
                        let mut uniform_s = (self.solver.chamber.scale.x + self.solver.chamber.scale.y + self.solver.chamber.scale.z) / 3.0;
                        if ui.add(egui::Slider::new(&mut uniform_s, 0.4..=3.0).text("Uniform Scale")).changed() {
                            self.solver.chamber.scale = Vec3::splat(uniform_s);
                            scale_changed = true;
                        }

                        // Independent X, Y, Z sliders
                        scale_changed |= ui.add(egui::Slider::new(&mut self.solver.chamber.scale.x, 0.3..=3.0).text("Scale X (Length)")).changed();
                        scale_changed |= ui.add(egui::Slider::new(&mut self.solver.chamber.scale.y, 0.3..=3.0).text("Scale Y (Height)")).changed();
                        scale_changed |= ui.add(egui::Slider::new(&mut self.solver.chamber.scale.z, 0.3..=3.0).text("Scale Z (Width)")).changed();

                        if ui.button("↺ Reset Scale (1.0x)").clicked() {
                            self.solver.chamber.scale = Vec3::ONE;
                            scale_changed = true;
                        }

                        if self.solver.chamber.shape == ChamberShape::Bathtub {
                            ui.separator();
                            ui.label("Bathtub Geometry:");
                            ui.add(egui::Slider::new(&mut self.solver.chamber.tub_slope_ratio, 0.1..=0.45).text("Backrest Slope"));
                            ui.add(egui::Slider::new(&mut self.solver.chamber.tub_fillet_ratio, 0.05..=0.35).text("Bottom Fillet"));
                        }

                        if scale_changed {
                            self.solver.bounds = self.solver.chamber.compute_bounds();
                            if let Some(ref mut gpu) = self.gpu_solver {
                                gpu.init_buffers(self.solver.bounds, &self.solver.config, &self.solver.particles);
                            }
                        }

                        let sz = self.solver.bounds.size();
                        ui.label(format!("Chamber Size: {:.2} x {:.2} x {:.2} m", sz.x, sz.y, sz.z));
                    });
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

                    // --- 1. Multi-Obstacles & .OBJ Models Controls ---
                    let obs_title = format!("🪨 Obstacles & .OBJ Models ({})", self.solver.colliders.len());
                    ui.collapsing(obs_title, |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("+ ⚪ Sphere").clicked() {
                                let id = self.next_collider_id;
                                self.next_collider_id += 1;
                                let mut c = Collider::sphere(Vec3::new(0.0, 0.45, 0.0), 0.18);
                                c.id = id;
                                c.name = format!("Sphere #{}", id);
                                self.solver.colliders.push(c);
                            }
                            if ui.button("+ 📦 Box").clicked() {
                                let id = self.next_collider_id;
                                self.next_collider_id += 1;
                                let mut c = Collider::box_obstacle(
                                    Vec3::new(-0.2, 0.05, -0.2),
                                    Vec3::new(0.2, 0.35, 0.2),
                                );
                                c.id = id;
                                c.name = format!("Box #{}", id);
                                self.solver.colliders.push(c);
                            }
                            if ui.button("+ 🥫 Cylinder").clicked() {
                                let id = self.next_collider_id;
                                self.next_collider_id += 1;
                                let mut c = Collider::cylinder(Vec3::new(0.0, 0.45, 0.0), 0.15, 0.4);
                                c.id = id;
                                c.name = format!("Cylinder #{}", id);
                                self.solver.colliders.push(c);
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("OBJ:");
                            ui.text_edit_singleline(&mut self.obj_file_path);
                            if ui.button("📥 Load .OBJ").clicked() {
                                let path = self.obj_file_path.trim().to_string();
                                match Collider::load_obj(&path, Vec3::new(0.0, 0.5, 0.0), 0.4) {
                                    Ok(mut col) => {
                                        let id = self.next_collider_id;
                                        self.next_collider_id += 1;
                                        col.id = id;
                                        let tri_count = if let ColliderShape::Mesh(ref m) = col.shape {
                                            m.triangles.len()
                                        } else {
                                            0
                                        };
                                        self.status_message = format!("Loaded OBJ: {} ({} tris)", path, tri_count);
                                        self.solver.colliders.push(col);
                                    }
                                    Err(e) => {
                                        self.status_message = format!("OBJ load error: {}", e);
                                    }
                                }
                            }
                        });

                        if !self.solver.colliders.is_empty() {
                            if ui.button("🗑 Clear All Obstacles").clicked() {
                                self.solver.colliders.clear();
                            }
                        }

                        let mut to_remove_col: Option<usize> = None;
                        for (idx, collider) in self.solver.colliders.iter_mut().enumerate() {
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut collider.enabled, "");
                                ui.label(egui::RichText::new(&collider.name).strong());
                                if ui.button("🗑").clicked() {
                                    to_remove_col = Some(idx);
                                }
                            });

                            if collider.enabled {
                                match &mut collider.shape {
                                    ColliderShape::Sphere { center, radius } => {
                                        ui.horizontal(|ui| {
                                            ui.label("Pos X,Y,Z:");
                                            ui.add(egui::DragValue::new(&mut center.x).speed(0.01).prefix("X: "));
                                            ui.add(egui::DragValue::new(&mut center.y).speed(0.01).prefix("Y: "));
                                            ui.add(egui::DragValue::new(&mut center.z).speed(0.01).prefix("Z: "));
                                        });
                                        ui.add(egui::Slider::new(radius, 0.02..=0.8).text("Radius"));
                                    }
                                    ColliderShape::Box { aabb } => {
                                        let mut center = aabb.center();
                                        let mut extents = aabb.size();
                                        let mut changed = false;
                                        ui.horizontal(|ui| {
                                            ui.label("Center:");
                                            changed |= ui.add(egui::DragValue::new(&mut center.x).speed(0.01).prefix("X: ")).changed();
                                            changed |= ui.add(egui::DragValue::new(&mut center.y).speed(0.01).prefix("Y: ")).changed();
                                            changed |= ui.add(egui::DragValue::new(&mut center.z).speed(0.01).prefix("Z: ")).changed();
                                        });
                                        ui.horizontal(|ui| {
                                            ui.label("Size:");
                                            changed |= ui.add(egui::DragValue::new(&mut extents.x).speed(0.01).prefix("W: ")).changed();
                                            changed |= ui.add(egui::DragValue::new(&mut extents.y).speed(0.01).prefix("H: ")).changed();
                                            changed |= ui.add(egui::DragValue::new(&mut extents.z).speed(0.01).prefix("D: ")).changed();
                                        });
                                        if changed {
                                            let h = extents * 0.5;
                                            *aabb = Aabb::new(center - h, center + h);
                                        }
                                    }
                                    ColliderShape::Cylinder { center, radius, height } => {
                                        ui.horizontal(|ui| {
                                            ui.label("Pos X,Y,Z:");
                                            ui.add(egui::DragValue::new(&mut center.x).speed(0.01).prefix("X: "));
                                            ui.add(egui::DragValue::new(&mut center.y).speed(0.01).prefix("Y: "));
                                            ui.add(egui::DragValue::new(&mut center.z).speed(0.01).prefix("Z: "));
                                        });
                                        ui.add(egui::Slider::new(radius, 0.02..=0.6).text("Radius"));
                                        ui.add(egui::Slider::new(height, 0.05..=1.2).text("Height"));
                                    }
                                    ColliderShape::Mesh(mesh_arc) => {
                                        if let Some(mesh) = std::sync::Arc::get_mut(mesh_arc) {
                                            ui.horizontal(|ui| {
                                                ui.label("Pos X,Y,Z:");
                                                let mut changed = false;
                                                changed |= ui.add(egui::DragValue::new(&mut mesh.position.x).speed(0.01).prefix("X: ")).changed();
                                                changed |= ui.add(egui::DragValue::new(&mut mesh.position.y).speed(0.01).prefix("Y: ")).changed();
                                                changed |= ui.add(egui::DragValue::new(&mut mesh.position.z).speed(0.01).prefix("Z: ")).changed();
                                                if changed {
                                                    mesh.update_transform();
                                                }
                                            });
                                            if ui.add(egui::Slider::new(&mut mesh.scale, 0.02..=2.5).text("Scale")).changed() {
                                                mesh.update_transform();
                                            }
                                            ui.horizontal(|ui| {
                                                ui.label("Rot:");
                                                let mut changed = false;
                                                changed |= ui.add(egui::DragValue::new(&mut mesh.rotation.x).speed(0.05).prefix("P: ")).changed();
                                                changed |= ui.add(egui::DragValue::new(&mut mesh.rotation.y).speed(0.05).prefix("Y: ")).changed();
                                                changed |= ui.add(egui::DragValue::new(&mut mesh.rotation.z).speed(0.05).prefix("R: ")).changed();
                                                if changed {
                                                    mesh.update_transform();
                                                }
                                            });
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(idx) = to_remove_col {
                            self.solver.colliders.remove(idx);
                        }
                    });

                    ui.separator();

                    // --- 2. Faucets & Inflow (Waterfall, Shower, Tap) ---
                    let faucet_title = format!("🚿 Faucets & Inflow ({})", self.solver.faucets.len());
                    ui.collapsing(faucet_title, |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("+ 🌊 Waterfall").clicked() {
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
                            if ui.button("+ 🚿 Shower").clicked() {
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
                            if ui.button("+ 🚰 Tap / Pipe").clicked() {
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

                        if !self.solver.faucets.is_empty() {
                            if ui.button("🗑 Clear All Faucets").clicked() {
                                self.solver.faucets.clear();
                            }
                        }

                        let mut to_remove_faucet: Option<usize> = None;
                        for (idx, faucet) in self.solver.faucets.iter_mut().enumerate() {
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut faucet.enabled, "");
                                ui.label(egui::RichText::new(&faucet.name).strong());
                                if ui.button("🗑").clicked() {
                                    to_remove_faucet = Some(idx);
                                }
                            });

                            if faucet.enabled {
                                ui.horizontal(|ui| {
                                    ui.label("Pos X,Y,Z:");
                                    ui.add(egui::DragValue::new(&mut faucet.position.x).speed(0.01).prefix("X: "));
                                    ui.add(egui::DragValue::new(&mut faucet.position.y).speed(0.01).prefix("Y: "));
                                    ui.add(egui::DragValue::new(&mut faucet.position.z).speed(0.01).prefix("Z: "));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Velocity:");
                                    ui.add(egui::DragValue::new(&mut faucet.direction.x).speed(0.05).prefix("Vx: "));
                                    ui.add(egui::DragValue::new(&mut faucet.direction.y).speed(0.05).prefix("Vy: "));
                                    ui.add(egui::DragValue::new(&mut faucet.direction.z).speed(0.05).prefix("Vz: "));
                                });
                                ui.add(egui::Slider::new(&mut faucet.flow_rate, 50.0..=2500.0).text("Flow Rate (p/s)"));

                                match &mut faucet.shape {
                                    FaucetShape::Circle { radius } => {
                                        ui.add(egui::Slider::new(radius, 0.02..=0.35).text("Nozzle Radius"));
                                        ui.add(egui::Slider::new(&mut faucet.spray_spread, 0.0..=0.5).text("Spray Angle"));
                                    }
                                    FaucetShape::Rectangle { width, depth } => {
                                        ui.add(egui::Slider::new(width, 0.05..=0.8).text("Weir Width"));
                                        ui.add(egui::Slider::new(depth, 0.02..=0.3).text("Weir Depth"));
                                    }
                                    FaucetShape::Point => {}
                                }
                            }
                        }
                        if let Some(idx) = to_remove_faucet {
                            self.solver.faucets.remove(idx);
                        }
                    });

                    ui.separator();

                    // --- 3. Drains & Sinks ---
                    let drain_title = format!("🕳️ Drains & Sinks ({})", self.solver.drains.len());
                    ui.collapsing(drain_title, |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("+ 🔘 Floor Drain").clicked() {
                                let id = self.next_drain_id;
                                self.next_drain_id += 1;
                                let d = Drain::new(
                                    format!("Floor Drain #{}", id),
                                    CubeFace::Floor,
                                    Vec3::new(0.0, 0.0, 0.0),
                                    0.18,
                                );
                                self.solver.drains.push(d);
                            }
                            if ui.button("+ 🚪 Wall Drain").clicked() {
                                let id = self.next_drain_id;
                                self.next_drain_id += 1;
                                let d = Drain::new(
                                    format!("Wall Drain #{}", id),
                                    CubeFace::Left,
                                    Vec3::new(0.0, 0.2, 0.0),
                                    0.15,
                                );
                                self.solver.drains.push(d);
                            }
                        });

                        if !self.solver.drains.is_empty() {
                            if ui.button("🗑 Clear All Drains").clicked() {
                                self.solver.drains.clear();
                            }
                        }

                        let mut to_remove_drain: Option<usize> = None;
                        for (idx, drain) in self.solver.drains.iter_mut().enumerate() {
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut drain.enabled, "");
                                ui.label(egui::RichText::new(&drain.name).strong());
                                if ui.button("🗑").clicked() {
                                    to_remove_drain = Some(idx);
                                }
                            });

                            if drain.enabled {
                                ui.horizontal(|ui| {
                                    ui.label("Face:");
                                    for &f in CubeFace::ALL {
                                        if ui.selectable_label(drain.face == f, f.name()).clicked() {
                                            drain.face = f;
                                        }
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Pos X,Y,Z:");
                                    ui.add(egui::DragValue::new(&mut drain.position.x).speed(0.01).prefix("X: "));
                                    ui.add(egui::DragValue::new(&mut drain.position.y).speed(0.01).prefix("Y: "));
                                    ui.add(egui::DragValue::new(&mut drain.position.z).speed(0.01).prefix("Z: "));
                                });
                                ui.add(egui::Slider::new(&mut drain.radius, 0.04..=0.5).text("Aperture Radius"));
                            }
                        }
                        if let Some(idx) = to_remove_drain {
                            self.solver.drains.remove(idx);
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
            let chamber = self.solver.chamber.clone();
            let colliders = self.solver.colliders.clone();
            let faucets = self.solver.faucets.clone();
            let drains = self.solver.drains.clone();
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
                            &chamber,
                            bounds,
                            &colliders,
                            &faucets,
                            &drains,
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
