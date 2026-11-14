use crate::math::Aabb;
use crate::render::{
    Camera, ColorMapType, ColorMode, EnvironmentPreset, FluidMaterialPreset, MeshTextureMode,
    RenderMode, SurfaceMeshConfig,
};
use crate::sph::chamber::{Chamber, ChamberShape};
use crate::sph::collider::{Collider, ColliderShape};
use crate::sph::faucet::{CubeFace, Drain, Faucet, FaucetShape};
use crate::sph::{FluidPhysicsPreset, ScenePreset, SimulationQuality, SolverConfig};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Portable JSON studio scene file (`.json`) shared between the GUI and CLI (`--scene` / `--save-scene`).
/// Uses `#[serde(default)]` on every struct so minimal hand-authored JSON snippets also load cleanly.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneConfigFile {
    pub version: u32,
    pub preset: String,
    pub quality: String,
    pub fluid_preset: String,
    pub is_2d: bool,
    pub sim_speed: f32,
    pub mouse_strength: f32,
    pub solver: SolverConfigDto,
    pub chamber: ChamberDto,
    pub colliders: Vec<ColliderDto>,
    pub faucets: Vec<FaucetDto>,
    pub drains: Vec<DrainDto>,
    pub camera: CameraDto,
    pub visuals: VisualsDto,
}

impl Default for SceneConfigFile {
    fn default() -> Self {
        Self {
            version: 1,
            preset: ScenePreset::DamBreak.slug().to_string(),
            quality: SimulationQuality::Medium.slug().to_string(),
            fluid_preset: FluidPhysicsPreset::Water.slug().to_string(),
            is_2d: false,
            sim_speed: 1.0,
            mouse_strength: 12.0,
            solver: SolverConfigDto::default(),
            chamber: ChamberDto::default(),
            colliders: Vec::new(),
            faucets: Vec::new(),
            drains: Vec::new(),
            camera: CameraDto::default(),
            visuals: VisualsDto::default(),
        }
    }
}

impl SceneConfigFile {
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
            }
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize scene JSON: {}", e))?;
        fs::write(path, json)
            .map_err(|e| format!("Failed to write scene file {}: {}", path.display(), e))
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read scene file {}: {}", path.display(), e))?;
        serde_json::from_str(&text)
            .map_err(|e| format!("Invalid scene JSON in {}: {}", path.display(), e))
    }

    pub fn parsed_preset(&self) -> ScenePreset {
        ScenePreset::from_slug(&self.preset).unwrap_or(ScenePreset::DamBreak)
    }

    pub fn parsed_quality(&self) -> SimulationQuality {
        SimulationQuality::from_slug(&self.quality).unwrap_or(SimulationQuality::Medium)
    }

    pub fn parsed_fluid_preset(&self) -> FluidPhysicsPreset {
        FluidPhysicsPreset::from_slug(&self.fluid_preset).unwrap_or(FluidPhysicsPreset::Custom)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverConfigDto {
    pub particle_radius: f32,
    pub spacing: f32,
    pub smoothing_length: f32,
    pub rest_density: f32,
    pub solver_iterations: usize,
    pub substeps: usize,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub vorticity: f32,
    pub air_drag: f32,
    pub restitution: f32,
    pub boundary_friction: f32,
    pub gravity: [f32; 3],
    pub tensile_k: f32,
    pub max_velocity: f32,
    pub wave_amplitude: f32,
    pub wave_frequency: f32,
}

impl Default for SolverConfigDto {
    fn default() -> Self {
        Self::from_config(&SolverConfig::default())
    }
}

impl SolverConfigDto {
    pub fn from_config(c: &SolverConfig) -> Self {
        Self {
            particle_radius: c.particle_radius,
            spacing: c.spacing,
            smoothing_length: c.smoothing_length,
            rest_density: c.rest_density,
            solver_iterations: c.solver_iterations,
            substeps: c.substeps,
            viscosity: c.viscosity,
            surface_tension: c.surface_tension,
            vorticity: c.vorticity,
            air_drag: c.air_drag,
            restitution: c.restitution,
            boundary_friction: c.boundary_friction,
            gravity: c.gravity.to_array(),
            tensile_k: c.tensile_k,
            max_velocity: c.max_velocity,
            wave_amplitude: c.wave_amplitude,
            wave_frequency: c.wave_frequency,
        }
    }

    pub fn to_config(&self) -> SolverConfig {
        SolverConfig {
            particle_radius: self.particle_radius.max(0.0005),
            spacing: self.spacing.max(0.001),
            smoothing_length: self.smoothing_length.max(0.002),
            rest_density: self.rest_density.max(1.0),
            solver_iterations: self.solver_iterations.clamp(1, 32),
            substeps: self.substeps.clamp(1, 32),
            viscosity: self.viscosity.clamp(0.0, 1.0),
            surface_tension: self.surface_tension.clamp(0.0, 1.0),
            vorticity: self.vorticity.clamp(0.0, 1.0),
            air_drag: self.air_drag.clamp(0.0, 5.0),
            restitution: self.restitution.clamp(0.0, 1.0),
            boundary_friction: self.boundary_friction.clamp(0.0, 1.0),
            gravity: Vec3::from_array(self.gravity),
            tensile_k: self.tensile_k.clamp(0.0, 0.05),
            max_velocity: self.max_velocity.clamp(1.0, 100.0),
            wave_amplitude: self.wave_amplitude.clamp(0.0, 2.0),
            wave_frequency: self.wave_frequency.clamp(0.05, 10.0),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ChamberDto {
    pub shape: String,
    pub scale: [f32; 3],
    pub base_size: [f32; 3],
    pub custom_size: bool,
    pub tub_slope_ratio: f32,
    pub tub_fillet_ratio: f32,
}

impl Default for ChamberDto {
    fn default() -> Self {
        Self::from_chamber(&Chamber::default())
    }
}

impl ChamberDto {
    pub fn from_chamber(c: &Chamber) -> Self {
        Self {
            shape: c.shape.slug().to_string(),
            scale: c.scale.to_array(),
            base_size: c.base_size.to_array(),
            custom_size: c.custom_size,
            tub_slope_ratio: c.tub_slope_ratio,
            tub_fillet_ratio: c.tub_fillet_ratio,
        }
    }

    pub fn to_chamber(&self) -> Chamber {
        Chamber {
            shape: ChamberShape::from_slug(&self.shape).unwrap_or(ChamberShape::Box),
            scale: Vec3::from_array(self.scale).max(Vec3::splat(0.1)),
            base_size: Vec3::from_array(self.base_size).max(Vec3::splat(0.1)),
            custom_size: self.custom_size,
            tub_slope_ratio: self.tub_slope_ratio.clamp(0.0, 0.48),
            tub_fillet_ratio: self.tub_fillet_ratio.clamp(0.0, 0.45),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ColliderDto {
    pub name: String,
    pub enabled: bool,
    pub kind: String,
    pub velocity: [f32; 3],
    pub restitution: f32,
    pub friction: f32,
    pub center: [f32; 3],
    pub radius: f32,
    pub height: f32,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub obj_path: Option<String>,
    pub scale: f32,
    pub rotation: [f32; 3],
}

impl Default for ColliderDto {
    fn default() -> Self {
        Self {
            name: "Sphere Obstacle".to_string(),
            enabled: true,
            kind: "sphere".to_string(),
            velocity: [0.0, 0.0, 0.0],
            restitution: 0.02,
            friction: 0.95,
            center: [0.0, 0.3, 0.0],
            radius: 0.2,
            height: 0.5,
            min: [-0.2, 0.0, -0.2],
            max: [0.2, 0.4, 0.2],
            obj_path: None,
            scale: 1.0,
            rotation: [0.0, 0.0, 0.0],
        }
    }
}

impl ColliderDto {
    pub fn from_collider(c: &Collider) -> Self {
        let mut dto = Self {
            name: c.name.clone(),
            enabled: c.enabled,
            velocity: c.velocity.to_array(),
            restitution: c.restitution,
            friction: c.friction,
            ..Self::default()
        };
        match &c.shape {
            ColliderShape::Sphere { center, radius } => {
                dto.kind = "sphere".to_string();
                dto.center = center.to_array();
                dto.radius = *radius;
            }
            ColliderShape::Box { aabb } => {
                dto.kind = "box".to_string();
                dto.min = aabb.min.to_array();
                dto.max = aabb.max.to_array();
            }
            ColliderShape::Cylinder {
                center,
                radius,
                height,
            } => {
                dto.kind = "cylinder".to_string();
                dto.center = center.to_array();
                dto.radius = *radius;
                dto.height = *height;
            }
            ColliderShape::Mesh(m) => {
                dto.kind = "mesh".to_string();
                dto.obj_path = m.source_path.clone();
                dto.center = m.position.to_array();
                dto.scale = m.scale;
                dto.rotation = m.rotation.to_array();
            }
        }
        dto
    }

    pub fn to_collider(&self, id: u32) -> Option<Collider> {
        let mut col = match self.kind.trim().to_ascii_lowercase().as_str() {
            "sphere" => Collider::sphere(Vec3::from_array(self.center), self.radius.max(0.01)),
            "box" => {
                let a = Vec3::from_array(self.min);
                let b = Vec3::from_array(self.max);
                Collider::box_obstacle(a.min(b), a.max(b))
            }
            "cylinder" => Collider::cylinder(
                Vec3::from_array(self.center),
                self.radius.max(0.01),
                self.height.max(0.01),
            ),
            "mesh" | "obj" => {
                let path = self.obj_path.as_deref()?;
                Collider::load_obj_transformed(
                    path,
                    Vec3::from_array(self.center),
                    self.scale.max(0.01),
                    Vec3::from_array(self.rotation),
                )
                .ok()?
            }
            _ => return None,
        };
        col.id = id;
        if !self.name.trim().is_empty() {
            col.name = self.name.clone();
        }
        col.enabled = self.enabled;
        col.velocity = Vec3::from_array(self.velocity);
        col.restitution = self.restitution.clamp(0.0, 1.0);
        col.friction = self.friction.clamp(0.0, 1.0);
        col.shape = match col.shape {
            ColliderShape::Box { aabb } => ColliderShape::Box {
                aabb: Aabb::new(aabb.min, aabb.max),
            },
            other => other,
        };
        Some(col)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FaucetDto {
    pub name: String,
    pub enabled: bool,
    pub kind: String,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub radius: f32,
    pub width: f32,
    pub depth: f32,
    pub flow_rate: f32,
    pub spray_spread: f32,
}

impl Default for FaucetDto {
    fn default() -> Self {
        Self {
            name: "Tap #1".to_string(),
            enabled: true,
            kind: "circle".to_string(),
            position: [0.0, 0.9, 0.0],
            direction: [0.0, -3.0, 0.0],
            radius: 0.12,
            width: 0.5,
            depth: 0.08,
            flow_rate: 1200.0,
            spray_spread: 0.0,
        }
    }
}

impl FaucetDto {
    pub fn from_faucet(f: &Faucet) -> Self {
        let mut dto = Self {
            name: f.name.clone(),
            enabled: f.enabled,
            position: f.position.to_array(),
            direction: f.direction.to_array(),
            flow_rate: f.flow_rate,
            spray_spread: f.spray_spread,
            ..Self::default()
        };
        match f.shape {
            FaucetShape::Circle { radius } => {
                dto.kind = "circle".to_string();
                dto.radius = radius;
            }
            FaucetShape::Rectangle { width, depth } => {
                dto.kind = "rectangle".to_string();
                dto.width = width;
                dto.depth = depth;
            }
            FaucetShape::Point => {
                dto.kind = "point".to_string();
            }
        }
        dto
    }

    pub fn to_faucet(&self, id: u32) -> Faucet {
        let pos = Vec3::from_array(self.position);
        let dir = Vec3::from_array(self.direction);
        let mut f = match self.kind.trim().to_ascii_lowercase().as_str() {
            "rectangle" | "waterfall" | "rect" => Faucet::new_waterfall(
                self.name.clone(),
                pos,
                self.width.max(0.01),
                self.depth.max(0.005),
                self.flow_rate.max(0.0),
            ),
            "point" | "jet" => Faucet::new_point(self.name.clone(), pos, dir, self.flow_rate.max(0.0)),
            _ => Faucet::new_circle(
                self.name.clone(),
                pos,
                dir,
                self.radius.max(0.005),
                self.flow_rate.max(0.0),
            ),
        };
        f.id = id;
        f.direction = dir;
        f.spray_spread = self.spray_spread.clamp(0.0, 1.4);
        f.enabled = self.enabled;
        f
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DrainDto {
    pub name: String,
    pub enabled: bool,
    pub face: String,
    pub position: [f32; 3],
    pub radius: f32,
}

impl Default for DrainDto {
    fn default() -> Self {
        Self {
            name: "Drain #1".to_string(),
            enabled: true,
            face: CubeFace::Floor.slug().to_string(),
            position: [0.0, 0.0, 0.0],
            radius: 0.16,
        }
    }
}

impl DrainDto {
    pub fn from_drain(d: &Drain) -> Self {
        Self {
            name: d.name.clone(),
            enabled: d.enabled,
            face: d.face.slug().to_string(),
            position: d.position.to_array(),
            radius: d.radius,
        }
    }

    pub fn to_drain(&self, id: u32) -> Drain {
        let mut d = Drain::new(
            self.name.clone(),
            CubeFace::from_slug(&self.face).unwrap_or(CubeFace::Floor),
            Vec3::from_array(self.position),
            self.radius.max(0.01),
        );
        d.id = id;
        d.enabled = self.enabled;
        d
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraDto {
    pub target: [f32; 3],
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
    pub fov_deg: f32,
    pub orbit_speed_deg: f32,
}

impl Default for CameraDto {
    fn default() -> Self {
        let cam = Camera::default();
        Self {
            target: cam.target.to_array(),
            azimuth: cam.azimuth,
            elevation: cam.elevation,
            distance: cam.distance,
            fov_deg: cam.fov_y.to_degrees(),
            orbit_speed_deg: 0.0,
        }
    }
}

impl CameraDto {
    pub fn from_camera(cam: &Camera, orbit_speed_deg: f32) -> Self {
        Self {
            target: cam.target.to_array(),
            azimuth: cam.azimuth,
            elevation: cam.elevation,
            distance: cam.distance,
            fov_deg: cam.fov_y.to_degrees(),
            orbit_speed_deg,
        }
    }

    pub fn to_camera(&self) -> Camera {
        let mut cam = Camera::default();
        cam.target = Vec3::from_array(self.target);
        cam.azimuth = self.azimuth;
        cam.elevation = self
            .elevation
            .clamp(-85.0f32.to_radians(), 85.0f32.to_radians());
        cam.distance = self.distance.clamp(0.5, 30.0);
        cam.fov_y = self.fov_deg.clamp(15.0, 110.0).to_radians();
        cam
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VisualsDto {
    pub render_mode: String,
    pub color_mode: String,
    pub color_map: String,
    pub particle_scale: f32,
    pub particle_foam: bool,
    pub lit_spheres: bool,
    pub enable_culling: bool,
    pub mesh: SurfaceMeshDto,
}

impl Default for VisualsDto {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::Particles.slug().to_string(),
            color_mode: ColorMode::Velocity.slug().to_string(),
            color_map: ColorMapType::Ocean.slug().to_string(),
            particle_scale: 1.25,
            particle_foam: true,
            lit_spheres: true,
            enable_culling: true,
            mesh: SurfaceMeshDto::default(),
        }
    }
}

impl VisualsDto {
    pub fn parsed_render_mode(&self) -> RenderMode {
        RenderMode::from_slug(&self.render_mode)
    }

    pub fn parsed_color_mode(&self) -> ColorMode {
        ColorMode::from_slug(&self.color_mode).unwrap_or(ColorMode::Velocity)
    }

    pub fn parsed_color_map(&self) -> ColorMapType {
        ColorMapType::from_slug(&self.color_map).unwrap_or(ColorMapType::Turbo)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SurfaceMeshDto {
    pub preset: String,
    pub shallow_color: [f32; 3],
    pub deep_color: [f32; 3],
    pub opacity: f32,
    pub absorption: f32,
    pub foam_intensity: f32,
    pub texture_mode: String,
    pub texture_scale: f32,
    pub texture_strength: f32,
    pub ior: f32,
    pub roughness: f32,
    pub specular_intensity: f32,
    pub refraction_strength: f32,
    pub subsurface_scattering: f32,
    pub env_reflection: f32,
    pub metallic: f32,
    pub light_dir: [f32; 3],
    pub light_intensity: f32,
    pub ambient_intensity: f32,
    pub cast_shadows: bool,
    pub voxel_scale: f32,
    pub iso_value: f32,
    pub smoothing_passes: usize,
    pub kernel_scale: f32,
    pub env_preset: String,
    pub exposure: f32,
    pub show_wireframe: bool,
    pub show_grid: bool,
}

impl Default for SurfaceMeshDto {
    fn default() -> Self {
        Self::from_config(&SurfaceMeshConfig::default())
    }
}

impl SurfaceMeshDto {
    pub fn from_config(c: &SurfaceMeshConfig) -> Self {
        Self {
            preset: c.preset.slug().to_string(),
            shallow_color: c.shallow_color,
            deep_color: c.deep_color,
            opacity: c.opacity,
            absorption: c.absorption,
            foam_intensity: c.foam_intensity,
            texture_mode: c.texture_mode.slug().to_string(),
            texture_scale: c.texture_scale,
            texture_strength: c.texture_strength,
            ior: c.ior,
            roughness: c.roughness,
            specular_intensity: c.specular_intensity,
            refraction_strength: c.refraction_strength,
            subsurface_scattering: c.subsurface_scattering,
            env_reflection: c.env_reflection,
            metallic: c.metallic,
            light_dir: c.light_dir.to_array(),
            light_intensity: c.light_intensity,
            ambient_intensity: c.ambient_intensity,
            cast_shadows: c.cast_shadows,
            voxel_scale: c.voxel_scale,
            iso_value: c.iso_value,
            smoothing_passes: c.smoothing_passes,
            kernel_scale: c.kernel_scale,
            env_preset: c.env_preset.slug().to_string(),
            exposure: c.exposure,
            show_wireframe: c.show_wireframe,
            show_grid: c.show_grid,
        }
    }

    pub fn to_config(&self) -> SurfaceMeshConfig {
        let mut cfg = SurfaceMeshConfig::default();
        cfg.preset = FluidMaterialPreset::from_slug(&self.preset);
        cfg.shallow_color = self.shallow_color;
        cfg.deep_color = self.deep_color;
        cfg.opacity = self.opacity.clamp(0.05, 1.0);
        cfg.absorption = self.absorption.clamp(0.0, 12.0);
        cfg.foam_intensity = self.foam_intensity.clamp(0.0, 1.0);
        cfg.texture_mode = MeshTextureMode::from_slug(&self.texture_mode);
        cfg.texture_scale = self.texture_scale.clamp(0.2, 64.0);
        cfg.texture_strength = self.texture_strength.clamp(0.0, 1.0);
        cfg.ior = self.ior.clamp(1.0, 3.0);
        cfg.roughness = self.roughness.clamp(0.01, 1.0);
        cfg.specular_intensity = self.specular_intensity.clamp(0.0, 4.0);
        cfg.refraction_strength = self.refraction_strength.clamp(0.0, 2.0);
        cfg.subsurface_scattering = self.subsurface_scattering.clamp(0.0, 2.0);
        cfg.env_reflection = self.env_reflection.clamp(0.0, 3.0);
        cfg.metallic = self.metallic.clamp(0.0, 1.0);
        let ld = Vec3::from_array(self.light_dir);
        cfg.light_dir = if ld.length_squared() > 1e-6 {
            ld.normalize()
        } else {
            Vec3::new(0.42, 0.82, 0.38).normalize()
        };
        cfg.light_intensity = self.light_intensity.clamp(0.0, 6.0);
        cfg.ambient_intensity = self.ambient_intensity.clamp(0.0, 3.0);
        cfg.cast_shadows = self.cast_shadows;
        cfg.voxel_scale = self.voxel_scale.clamp(0.25, 2.0);
        cfg.iso_value = self.iso_value.clamp(0.08, 0.90);
        cfg.smoothing_passes = self.smoothing_passes.min(24);
        cfg.kernel_scale = self.kernel_scale.clamp(0.5, 2.5);
        cfg.env_preset =
            EnvironmentPreset::from_slug(&self.env_preset).unwrap_or(EnvironmentPreset::StudioDark);
        cfg.exposure = self.exposure.clamp(0.2, 3.5);
        cfg.show_wireframe = self.show_wireframe;
        cfg.show_grid = self.show_grid;
        cfg
    }
}
