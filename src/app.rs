use crate::export::PlyExporter;
use crate::math::{unproject_mouse_ray, Aabb};
use crate::render::{Camera, ColorMap, ColorMapType, ColorMode, GlRenderer, ParticleVertex};
use crate::sph::collider::Collider;
use crate::sph::scene::{SceneGenerator, ScenePreset};
use crate::sph::solver::{SolverConfig, SphSolver};
use eframe::egui;
use glam::Vec3;
use glow::HasContext as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct LiquidApp {
    pub solver: SphSolver,
    pub camera: Camera,
    pub is_playing: bool,
    pub fixed_dt: f32,
    pub current_preset: ScenePreset,

    // Rendering settings
    pub renderer: Arc<Mutex<Option<GlRenderer>>>,
    pub lit_spheres: bool,
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
}

impl LiquidApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let bounds = Aabb::new(Vec3::new(-1.0, 0.0, -0.6), Vec3::new(1.0, 1.8, 0.6));
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

        Self {
            solver,
            camera: Camera::default(),
            is_playing: true,
            fixed_dt: 0.016,
            current_preset: preset,
            renderer,
            lit_spheres: true,
            color_mode: ColorMode::Velocity,
            color_map: ColorMapType::Ocean,
            display_size_factor: 1.0,
            sphere_obstacle_pos: Vec3::new(0.0, 0.6, 0.0),
            sphere_obstacle_radius: 0.28,
            sphere_obstacle_enabled: false,
            last_frame_time: Instant::now(),
            frame_counter: 0,
            current_fps: 60.0,
            last_step_time_ms: 0.0,
            export_path: "exports".to_string(),
            export_sequence: false,
            exported_frames_count: 0,
            status_message: "Ready".to_string(),
        }
    }

    pub fn load_preset(&mut self, preset: ScenePreset) {
        self.current_preset = preset;
        SceneGenerator::setup_scene(&mut self.solver, preset);
        if preset == ScenePreset::ObstacleDrop {
            self.sphere_obstacle_enabled = true;
        }
        self.status_message = format!("Loaded preset: {}", preset.name());
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
            ColorMode::Velocity => (p.speed() / 4.0).clamp(0.0, 1.0),
            ColorMode::Pressure => (p.pressure / 1500.0).clamp(0.0, 1.0),
            ColorMode::Density => {
                let dev = (p.density - self.solver.config.rest_density).abs()
                    / self.solver.config.rest_density;
                (dev * 6.0).clamp(0.0, 1.0)
            }
            ColorMode::Height => {
                let y = p.position.y;
                let b = &self.solver.bounds;
                ((y - b.min.y) / (b.max.y - b.min.y)).clamp(0.0, 1.0)
            }
            ColorMode::KineticEnergy => (p.kinetic_energy() / 10.0).clamp(0.0, 1.0),
            ColorMode::SingleColor => return [50, 150, 230],
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
            self.solver.step(self.fixed_dt);
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
                    self.solver.step(self.fixed_dt);
                    self.last_step_time_ms = start.elapsed().as_secs_f32() * 1000.0;
                    ctx.request_repaint();
                }

                if ui.button("🔄 Reset").clicked() {
                    self.load_preset(self.current_preset);
                }

                ui.separator();

                // Quick preset buttons
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
                ui.label(format!("Particles: {}", self.solver.particle_count()));
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

                    // --- Physics Parameters ---
                    ui.collapsing("⚙ Fluid Physics", |ui| {
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.viscosity, 0.005..=1.5)
                                .text("Viscosity")
                                .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.stiffness, 200.0..=5000.0)
                                .text("Stiffness (k)"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.solver.config.surface_tension,
                                0.0..=0.3,
                            )
                            .text("Surface Tension"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.gravity.y, -25.0..=5.0)
                                .text("Gravity Y"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.solver.config.boundary_restitution,
                                0.0..=0.9,
                            )
                            .text("Wall Restitution"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.solver.config.boundary_friction,
                                0.5..=1.0,
                            )
                            .text("Wall Friction"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.solver.config.substeps, 1..=8)
                                .text("Substeps"),
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
                                    egui::Slider::new(&mut self.sphere_obstacle_pos.x, -0.8..=0.8)
                                        .text("Pos X"),
                                )
                                .changed();
                            obstacle_changed |= ui
                                .add(
                                    egui::Slider::new(&mut self.sphere_obstacle_pos.y, 0.1..=1.4)
                                        .text("Pos Y"),
                                )
                                .changed();
                            obstacle_changed |= ui
                                .add(
                                    egui::Slider::new(&mut self.sphere_obstacle_pos.z, -0.6..=0.6)
                                        .text("Pos Z"),
                                )
                                .changed();
                            obstacle_changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.sphere_obstacle_radius,
                                        0.08..=0.5,
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
                        ui.add(
                            egui::Slider::new(&mut self.display_size_factor, 0.4..=2.5)
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

            // Prepare vertex data
            let particles_data: Vec<ParticleVertex> = self
                .solver
                .particles
                .iter()
                .map(|p| {
                    let c = self.compute_particle_color(p);
                    ParticleVertex {
                        pos: p.position.to_array(),
                        color: [
                            c[0] as f32 / 255.0,
                            c[1] as f32 / 255.0,
                            c[2] as f32 / 255.0,
                            0.95,
                        ],
                    }
                })
                .collect();

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
    }
}
