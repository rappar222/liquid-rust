use crate::sph::particle::{Particle, ParticleType};
use crate::sph::spatial::SpatialHashGrid;
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Selectable fluid rendering mode (supported in both GUI and CLI).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    Particles,
    SurfaceMesh,
}

impl RenderMode {
    pub const ALL: &'static [RenderMode] = &[RenderMode::Particles, RenderMode::SurfaceMesh];

    pub fn name(&self) -> &'static str {
        match self {
            RenderMode::Particles => "SPH Spheres (Particles)",
            RenderMode::SurfaceMesh => "3D Surface Mesh (Isosurface)",
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            RenderMode::Particles => "particles",
            RenderMode::SurfaceMesh => "mesh",
        }
    }

    pub fn from_slug(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "mesh" | "surface" | "surface-mesh" | "isosurface" | "marching-cubes" => {
                RenderMode::SurfaceMesh
            }
            _ => RenderMode::Particles,
        }
    }
}

/// Curated optical & physical material presets for surface-meshed liquids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluidMaterialPreset {
    ClearWater,
    DeepOcean,
    TropicalLagoon,
    MoltenAmber,
    CrimsonWine,
    EmeraldGel,
    Milk,
    LiquidChrome,
    Custom,
}

impl FluidMaterialPreset {
    pub const ALL: &'static [FluidMaterialPreset] = &[
        FluidMaterialPreset::ClearWater,
        FluidMaterialPreset::DeepOcean,
        FluidMaterialPreset::TropicalLagoon,
        FluidMaterialPreset::MoltenAmber,
        FluidMaterialPreset::CrimsonWine,
        FluidMaterialPreset::EmeraldGel,
        FluidMaterialPreset::Milk,
        FluidMaterialPreset::LiquidChrome,
        FluidMaterialPreset::Custom,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            FluidMaterialPreset::ClearWater => "Clear Water (Crystal)",
            FluidMaterialPreset::DeepOcean => "Deep Ocean (Atlantic)",
            FluidMaterialPreset::TropicalLagoon => "Tropical Lagoon (Turquoise)",
            FluidMaterialPreset::MoltenAmber => "Molten Amber (Honey Gold)",
            FluidMaterialPreset::CrimsonWine => "Crimson Wine (Ruby)",
            FluidMaterialPreset::EmeraldGel => "Emerald Gel (Jade)",
            FluidMaterialPreset::Milk => "Creamy Milk (Subsurface)",
            FluidMaterialPreset::LiquidChrome => "Liquid Chrome (Mercury)",
            FluidMaterialPreset::Custom => "Custom Material",
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            FluidMaterialPreset::ClearWater => "water",
            FluidMaterialPreset::DeepOcean => "ocean",
            FluidMaterialPreset::TropicalLagoon => "lagoon",
            FluidMaterialPreset::MoltenAmber => "amber",
            FluidMaterialPreset::CrimsonWine => "wine",
            FluidMaterialPreset::EmeraldGel => "emerald",
            FluidMaterialPreset::Milk => "milk",
            FluidMaterialPreset::LiquidChrome => "chrome",
            FluidMaterialPreset::Custom => "custom",
        }
    }

    pub fn from_slug(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "water" | "clear" | "clear-water" => FluidMaterialPreset::ClearWater,
            "ocean" | "deep-ocean" | "sea" => FluidMaterialPreset::DeepOcean,
            "lagoon" | "tropical" | "aqua" => FluidMaterialPreset::TropicalLagoon,
            "amber" | "honey" | "gold" | "lava" => FluidMaterialPreset::MoltenAmber,
            "wine" | "crimson" | "ruby" | "blood" => FluidMaterialPreset::CrimsonWine,
            "emerald" | "jade" | "slime" => FluidMaterialPreset::EmeraldGel,
            "milk" | "cream" => FluidMaterialPreset::Milk,
            "chrome" | "mercury" | "metal" | "silver" => FluidMaterialPreset::LiquidChrome,
            _ => FluidMaterialPreset::Custom,
        }
    }

    pub fn apply(&self, cfg: &mut SurfaceMeshConfig) {
        cfg.preset = *self;
        match self {
            FluidMaterialPreset::ClearWater => {
                cfg.shallow_color = [0.42, 0.86, 0.98];
                cfg.deep_color = [0.03, 0.24, 0.52];
                cfg.opacity = 0.62;
                cfg.absorption = 2.1;
                cfg.ior = 1.333;
                cfg.roughness = 0.045;
                cfg.specular_intensity = 1.45;
                cfg.refraction_strength = 0.60;
                cfg.subsurface_scattering = 0.45;
                cfg.env_reflection = 0.90;
                cfg.metallic = 0.0;
                cfg.foam_intensity = 0.35;
            }
            FluidMaterialPreset::DeepOcean => {
                cfg.shallow_color = [0.12, 0.66, 0.85];
                cfg.deep_color = [0.015, 0.09, 0.26];
                cfg.opacity = 0.78;
                cfg.absorption = 3.2;
                cfg.ior = 1.34;
                cfg.roughness = 0.065;
                cfg.specular_intensity = 1.35;
                cfg.refraction_strength = 0.50;
                cfg.subsurface_scattering = 0.55;
                cfg.env_reflection = 0.95;
                cfg.metallic = 0.0;
                cfg.foam_intensity = 0.60;
            }
            FluidMaterialPreset::TropicalLagoon => {
                cfg.shallow_color = [0.18, 0.94, 0.86];
                cfg.deep_color = [0.02, 0.32, 0.44];
                cfg.opacity = 0.68;
                cfg.absorption = 2.4;
                cfg.ior = 1.335;
                cfg.roughness = 0.05;
                cfg.specular_intensity = 1.40;
                cfg.refraction_strength = 0.58;
                cfg.subsurface_scattering = 0.65;
                cfg.env_reflection = 0.85;
                cfg.metallic = 0.0;
                cfg.foam_intensity = 0.42;
            }
            FluidMaterialPreset::MoltenAmber => {
                cfg.shallow_color = [1.00, 0.72, 0.16];
                cfg.deep_color = [0.42, 0.11, 0.01];
                cfg.opacity = 0.82;
                cfg.absorption = 3.0;
                cfg.ior = 1.49;
                cfg.roughness = 0.08;
                cfg.specular_intensity = 1.50;
                cfg.refraction_strength = 0.48;
                cfg.subsurface_scattering = 0.75;
                cfg.env_reflection = 0.80;
                cfg.metallic = 0.0;
                cfg.foam_intensity = 0.12;
            }
            FluidMaterialPreset::CrimsonWine => {
                cfg.shallow_color = [0.88, 0.14, 0.24];
                cfg.deep_color = [0.14, 0.008, 0.03];
                cfg.opacity = 0.86;
                cfg.absorption = 3.8;
                cfg.ior = 1.36;
                cfg.roughness = 0.04;
                cfg.specular_intensity = 1.55;
                cfg.refraction_strength = 0.45;
                cfg.subsurface_scattering = 0.40;
                cfg.env_reflection = 0.92;
                cfg.metallic = 0.0;
                cfg.foam_intensity = 0.10;
            }
            FluidMaterialPreset::EmeraldGel => {
                cfg.shallow_color = [0.24, 0.92, 0.45];
                cfg.deep_color = [0.02, 0.22, 0.09];
                cfg.opacity = 0.80;
                cfg.absorption = 2.8;
                cfg.ior = 1.42;
                cfg.roughness = 0.07;
                cfg.specular_intensity = 1.35;
                cfg.refraction_strength = 0.52;
                cfg.subsurface_scattering = 0.85;
                cfg.env_reflection = 0.78;
                cfg.metallic = 0.0;
                cfg.foam_intensity = 0.18;
            }
            FluidMaterialPreset::Milk => {
                cfg.shallow_color = [0.97, 0.96, 0.93];
                cfg.deep_color = [0.78, 0.76, 0.70];
                cfg.opacity = 0.96;
                cfg.absorption = 4.5;
                cfg.ior = 1.35;
                cfg.roughness = 0.18;
                cfg.specular_intensity = 0.95;
                cfg.refraction_strength = 0.10;
                cfg.subsurface_scattering = 1.10;
                cfg.env_reflection = 0.45;
                cfg.metallic = 0.0;
                cfg.foam_intensity = 0.05;
            }
            FluidMaterialPreset::LiquidChrome => {
                cfg.shallow_color = [0.86, 0.89, 0.94];
                cfg.deep_color = [0.55, 0.60, 0.68];
                cfg.opacity = 1.00;
                cfg.absorption = 6.0;
                cfg.ior = 2.40;
                cfg.roughness = 0.035;
                cfg.specular_intensity = 2.10;
                cfg.refraction_strength = 0.0;
                cfg.subsurface_scattering = 0.0;
                cfg.env_reflection = 1.65;
                cfg.metallic = 0.95;
                cfg.foam_intensity = 0.0;
            }
            FluidMaterialPreset::Custom => {}
        }
    }
}

/// Triplanar procedural surface texture modes for the reconstructed liquid mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshTextureMode {
    None,
    Ripples,
    Caustics,
    Foam,
    Grid,
}

impl MeshTextureMode {
    pub const ALL: &'static [MeshTextureMode] = &[
        MeshTextureMode::None,
        MeshTextureMode::Ripples,
        MeshTextureMode::Caustics,
        MeshTextureMode::Foam,
        MeshTextureMode::Grid,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            MeshTextureMode::None => "None (Smooth Glass)",
            MeshTextureMode::Ripples => "Capillary Ripples (Micro-Waves)",
            MeshTextureMode::Caustics => "Voronoi Caustics (Light Web)",
            MeshTextureMode::Foam => "Turbulent Foam (FBM Streaks)",
            MeshTextureMode::Grid => "Triplanar UV Grid (Contour)",
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            MeshTextureMode::None => "none",
            MeshTextureMode::Ripples => "ripples",
            MeshTextureMode::Caustics => "caustics",
            MeshTextureMode::Foam => "foam",
            MeshTextureMode::Grid => "grid",
        }
    }

    pub fn from_slug(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "smooth" | "flat" | "off" => MeshTextureMode::None,
            "ripples" | "waves" | "ripple" => MeshTextureMode::Ripples,
            "caustics" | "caustic" | "cellular" | "voronoi" => MeshTextureMode::Caustics,
            "foam" | "noise" | "fbm" => MeshTextureMode::Foam,
            "grid" | "checker" | "uv" => MeshTextureMode::Grid,
            _ => MeshTextureMode::Ripples,
        }
    }

    pub fn gpu_id(&self) -> i32 {
        match self {
            MeshTextureMode::None => 0,
            MeshTextureMode::Ripples => 1,
            MeshTextureMode::Caustics => 2,
            MeshTextureMode::Foam => 3,
            MeshTextureMode::Grid => 4,
        }
    }
}

/// Evaluated procedural texture sample at a surface point.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceTexSample {
    pub normal: Vec3,
    pub albedo_scale: Vec3,
    pub emissive_caustic: f32,
    pub foam_boost: f32,
}

/// Full configuration for surface meshing geometry, texture, color, and optical light behavior.
/// Offline CLI rendering quality tiers (strictly for headless CLI exports; never shown in interactive GUI).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CliRenderQuality {
    /// Standard 1x rasterization, 260-cell surface mesh cap, 128px shadow map, H.264 CRF 18.
    #[default]
    Standard,
    /// Studio offline quality: 2x2 SSAA, 384-cell mesh cap, 256px soft shadow map, H.264 CRF 15.
    Studio,
    /// Cinema offline quality: 3x3 SSAA, 512-cell mesh cap, 384px shadow map, dispersion, H.264 CRF 13.
    Cinema,
    /// Master offline quality: 4x4 SSAA, 640-cell mesh cap, 512px shadow map, dispersion + high smoothing, H.264 CRF 11.
    Master,
}

impl CliRenderQuality {
    pub const ALL: &'static [CliRenderQuality] = &[
        CliRenderQuality::Standard,
        CliRenderQuality::Studio,
        CliRenderQuality::Cinema,
        CliRenderQuality::Master,
    ];

    pub fn is_cli_only(&self) -> bool {
        !matches!(self, CliRenderQuality::Standard)
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "standard" | "default" | "1x" => Some(Self::Standard),
            "studio" | "photoreal" | "2x" => Some(Self::Studio),
            "cinema" | "film" | "3x" => Some(Self::Cinema),
            "master" | "hyper" | "ultra-cli" | "4x" => Some(Self::Master),
            _ => None,
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            CliRenderQuality::Standard => "standard",
            CliRenderQuality::Studio => "studio",
            CliRenderQuality::Cinema => "cinema",
            CliRenderQuality::Master => "master",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            CliRenderQuality::Standard => "Standard (1x)",
            CliRenderQuality::Studio => "Studio [CLI] (2x2 SSAA)",
            CliRenderQuality::Cinema => "Cinema [CLI] (3x3 SSAA)",
            CliRenderQuality::Master => "Master [CLI] (4x4 SSAA)",
        }
    }

    pub fn ssaa_factor(&self) -> usize {
        match self {
            CliRenderQuality::Standard => 1,
            CliRenderQuality::Studio => 2,
            CliRenderQuality::Cinema => 3,
            CliRenderQuality::Master => 4,
        }
    }

    pub fn shadow_res(&self) -> usize {
        match self {
            CliRenderQuality::Standard => 128,
            CliRenderQuality::Studio => 256,
            CliRenderQuality::Cinema => 384,
            CliRenderQuality::Master => 512,
        }
    }

    pub fn ffmpeg_preset(&self) -> &'static str {
        match self {
            CliRenderQuality::Standard => "medium",
            CliRenderQuality::Studio => "slow",
            CliRenderQuality::Cinema => "slow",
            CliRenderQuality::Master => "slower",
        }
    }

    pub fn ffmpeg_crf(&self) -> u32 {
        match self {
            CliRenderQuality::Standard => 18,
            CliRenderQuality::Studio => 15,
            CliRenderQuality::Cinema => 13,
            CliRenderQuality::Master => 11,
        }
    }

    pub fn chromatic_dispersion(&self) -> f32 {
        match self {
            CliRenderQuality::Standard => 0.0,
            CliRenderQuality::Studio => 0.008,
            CliRenderQuality::Cinema => 0.018,
            CliRenderQuality::Master => 0.028,
        }
    }

    pub fn apply_to_mesh_config(
        &self,
        cfg: &mut SurfaceMeshConfig,
        user_set_smooth: bool,
        user_set_voxel: bool,
    ) {
        match self {
            CliRenderQuality::Standard => {
                cfg.max_grid_cells = 260.0;
                cfg.min_voxel_size = 0.0025;
                cfg.max_smoothing_passes = 10;
            }
            CliRenderQuality::Studio => {
                cfg.max_grid_cells = 384.0;
                cfg.min_voxel_size = 0.0014;
                cfg.max_smoothing_passes = 14;
                if !user_set_smooth {
                    cfg.smoothing_passes = cfg.smoothing_passes.max(7);
                }
                if !user_set_voxel {
                    cfg.voxel_scale = cfg.voxel_scale.min(0.62);
                }
            }
            CliRenderQuality::Cinema => {
                cfg.max_grid_cells = 512.0;
                cfg.min_voxel_size = 0.0009;
                cfg.max_smoothing_passes = 18;
                if !user_set_smooth {
                    cfg.smoothing_passes = cfg.smoothing_passes.max(9);
                }
                if !user_set_voxel {
                    cfg.voxel_scale = cfg.voxel_scale.min(0.55);
                }
            }
            CliRenderQuality::Master => {
                cfg.max_grid_cells = 640.0;
                cfg.min_voxel_size = 0.0006;
                cfg.max_smoothing_passes = 24;
                if !user_set_smooth {
                    cfg.smoothing_passes = cfg.smoothing_passes.max(12);
                }
                if !user_set_voxel {
                    cfg.voxel_scale = cfg.voxel_scale.min(0.48);
                }
            }
        }
    }
}

/// Studio lighting & sky backdrop presets shared across OpenGL viewport and offline MP4/PNG renders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvironmentPreset {
    StudioDark,
    DaylightSky,
    GoldenSunset,
    ArcticLab,
    AbyssalDeep,
}

impl EnvironmentPreset {
    pub const ALL: &'static [EnvironmentPreset] = &[
        EnvironmentPreset::StudioDark,
        EnvironmentPreset::DaylightSky,
        EnvironmentPreset::GoldenSunset,
        EnvironmentPreset::ArcticLab,
        EnvironmentPreset::AbyssalDeep,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            EnvironmentPreset::StudioDark => "Studio Dark (Default)",
            EnvironmentPreset::DaylightSky => "Daylight Sky (Clear Noon)",
            EnvironmentPreset::GoldenSunset => "Golden Sunset (Warm Rim)",
            EnvironmentPreset::ArcticLab => "Arctic Lab (Clean Slate)",
            EnvironmentPreset::AbyssalDeep => "Abyssal Deep (Midnight)",
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            EnvironmentPreset::StudioDark => "studio",
            EnvironmentPreset::DaylightSky => "daylight",
            EnvironmentPreset::GoldenSunset => "sunset",
            EnvironmentPreset::ArcticLab => "arctic",
            EnvironmentPreset::AbyssalDeep => "abyssal",
        }
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "studio" | "dark" | "studio-dark" | "default" => Some(EnvironmentPreset::StudioDark),
            "daylight" | "day" | "sky" | "noon" => Some(EnvironmentPreset::DaylightSky),
            "sunset" | "golden" | "dusk" => Some(EnvironmentPreset::GoldenSunset),
            "arctic" | "lab" | "light" => Some(EnvironmentPreset::ArcticLab),
            "abyssal" | "deep" | "midnight" => Some(EnvironmentPreset::AbyssalDeep),
            _ => None,
        }
    }

    /// Background sky linear-space `(nadir_low, zenith_high)` gradient.
    pub fn bg_gradient_linear(&self) -> (Vec3, Vec3) {
        match self {
            EnvironmentPreset::StudioDark => {
                (Vec3::new(0.012, 0.016, 0.026), Vec3::new(0.036, 0.052, 0.082))
            }
            EnvironmentPreset::DaylightSky => {
                (Vec3::new(0.090, 0.150, 0.250), Vec3::new(0.035, 0.095, 0.240))
            }
            EnvironmentPreset::GoldenSunset => {
                (Vec3::new(0.110, 0.052, 0.042), Vec3::new(0.028, 0.036, 0.085))
            }
            EnvironmentPreset::ArcticLab => {
                (Vec3::new(0.065, 0.082, 0.110), Vec3::new(0.110, 0.145, 0.195))
            }
            EnvironmentPreset::AbyssalDeep => {
                (Vec3::new(0.003, 0.006, 0.014), Vec3::new(0.008, 0.018, 0.038))
            }
        }
    }

    /// Approximate sRGB `[0..255]` background color for the OpenGL viewport canvas.
    pub fn viewport_bg_rgb8(&self) -> [u8; 3] {
        match self {
            EnvironmentPreset::StudioDark => [11, 13, 19],
            EnvironmentPreset::DaylightSky => [24, 42, 72],
            EnvironmentPreset::GoldenSunset => [30, 19, 24],
            EnvironmentPreset::ArcticLab => [34, 42, 54],
            EnvironmentPreset::AbyssalDeep => [5, 8, 14],
        }
    }

    /// Linear `(horizon, zenith, ground)` reflection colors for dielectric/metallic Fresnel reflections.
    pub fn sky_reflection_linear(&self) -> (Vec3, Vec3, Vec3) {
        match self {
            EnvironmentPreset::StudioDark => (
                Vec3::new(0.24, 0.36, 0.52),
                Vec3::new(0.08, 0.16, 0.30),
                Vec3::new(0.025, 0.032, 0.045),
            ),
            EnvironmentPreset::DaylightSky => (
                Vec3::new(0.55, 0.74, 0.95),
                Vec3::new(0.14, 0.36, 0.76),
                Vec3::new(0.05, 0.07, 0.10),
            ),
            EnvironmentPreset::GoldenSunset => (
                Vec3::new(0.88, 0.44, 0.22),
                Vec3::new(0.16, 0.14, 0.34),
                Vec3::new(0.04, 0.025, 0.03),
            ),
            EnvironmentPreset::ArcticLab => (
                Vec3::new(0.62, 0.70, 0.80),
                Vec3::new(0.32, 0.42, 0.56),
                Vec3::new(0.08, 0.10, 0.13),
            ),
            EnvironmentPreset::AbyssalDeep => (
                Vec3::new(0.08, 0.20, 0.34),
                Vec3::new(0.02, 0.06, 0.14),
                Vec3::new(0.008, 0.012, 0.022),
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceMeshConfig {
    // 1. Meshing Geometry
    pub voxel_scale: f32,         // Cell size = particle_radius * voxel_scale (default 0.72)
    pub iso_value: f32,           // Scalar field isosurface threshold (default 0.42)
    pub smoothing_passes: usize,  // Volume-preserving Taubin smoothing iterations (default 3)
    pub kernel_scale: f32,        // Splat kernel support multiplier (default 1.0)

    // 2. Material & Color
    pub preset: FluidMaterialPreset,
    pub shallow_color: [f32; 3],  // Surface / thin-film color [0..1]
    pub deep_color: [f32; 3],     // Deep Beer-Lambert absorption color [0..1]
    pub opacity: f32,             // Base surface opacity [0.05..1.0]
    pub absorption: f32,          // Optical thickness density [0.0..8.0]
    pub foam_intensity: f32,      // Whitewater crest aeration [0.0..1.0]

    // 3. Procedural Texture
    pub texture_mode: MeshTextureMode,
    pub texture_scale: f32,       // Spatial frequency (default 6.0)
    pub texture_strength: f32,    // Bump & pattern intensity [0.0..1.0] (default 0.30)

    // 4. Light Behavior, Studio Environment & Optics
    pub ior: f32,                 // Index of refraction (1.333 for water)
    pub roughness: f32,           // GGX microfacet roughness [0.01..1.0]
    pub specular_intensity: f32,  // Direct sun specular highlight multiplier
    pub refraction_strength: f32, // Screen-space background refraction distortion
    pub subsurface_scattering: f32, // Back-lit thin wave crest glow
    pub env_reflection: f32,      // Studio sky / horizon Fresnel reflection
    pub metallic: f32,            // Metallic reflectance [0.0..1.0]
    pub light_dir: Vec3,          // Normalized key sun direction
    pub light_intensity: f32,     // Key light radiance
    pub ambient_intensity: f32,   // Hemisphere ambient illumination
    pub cast_shadows: bool,       // Floor shadow & caustic projection in CLI renders
    pub env_preset: EnvironmentPreset, // Sky & reflection environment preset
    pub exposure: f32,            // Filmic camera exposure multiplier (default 1.0)
    pub show_wireframe: bool,     // Render chamber/obstacle/emitter wireframes
    pub show_grid: bool,          // Render 3D studio floor grid
    // 5. Resolution & Smoothing Ceilings (raised for offline CLI modes)
    pub max_grid_cells: f32,        // Max marching-cubes cells along longest axis (default 260)
    pub min_voxel_size: f32,        // Minimum marching-cubes voxel size in meters (default 0.0025)
    pub max_smoothing_passes: usize,// Upper bound for Taubin smoothing iterations (default 10)
}

impl Default for SurfaceMeshConfig {
    fn default() -> Self {
        let mut cfg = Self {
            voxel_scale: 0.72,
            iso_value: 0.42,
            smoothing_passes: 5,
            kernel_scale: 1.0,
            preset: FluidMaterialPreset::ClearWater,
            shallow_color: [0.42, 0.86, 0.98],
            deep_color: [0.03, 0.24, 0.52],
            opacity: 0.62,
            absorption: 2.1,
            foam_intensity: 0.35,
            texture_mode: MeshTextureMode::Ripples,
            texture_scale: 6.0,
            texture_strength: 0.28,
            ior: 1.333,
            roughness: 0.045,
            specular_intensity: 1.45,
            refraction_strength: 0.60,
            subsurface_scattering: 0.45,
            env_reflection: 0.90,
            metallic: 0.0,
            light_dir: Vec3::new(0.42, 0.82, 0.38).normalize(),
            light_intensity: 1.25,
            ambient_intensity: 0.32,
            cast_shadows: true,
            env_preset: EnvironmentPreset::StudioDark,
            exposure: 1.0,
            show_wireframe: true,
            show_grid: true,
            max_grid_cells: 260.0,
            min_voxel_size: 0.0025,
            max_smoothing_passes: 10,
        };
        FluidMaterialPreset::ClearWater.apply(&mut cfg);
        cfg
    }
}

impl SurfaceMeshConfig {
    /// Parse `#RRGGBB`, `RRGGBB`, or `r,g,b` (either `0..=1` floats or `0..=255` ints) into `[f32; 3]`.
    pub fn parse_rgb(s: &str) -> Option<[f32; 3]> {
        let trimmed = s.trim();
        let hex = trimmed.strip_prefix('#').unwrap_or(trimmed);
        if hex.len() == 6 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()? as f32 / 255.0;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()? as f32 / 255.0;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()? as f32 / 255.0;
            return Some([r, g, b]);
        }

        let parts: Vec<f32> = trimmed
            .split(',')
            .filter_map(|p| p.trim().parse::<f32>().ok())
            .collect();
        if parts.len() == 3 {
            let scale = if parts.iter().any(|&v| v > 1.0) {
                1.0 / 255.0
            } else {
                1.0
            };
            return Some([
                (parts[0] * scale).clamp(0.0, 1.0),
                (parts[1] * scale).clamp(0.0, 1.0),
                (parts[2] * scale).clamp(0.0, 1.0),
            ]);
        }
        None
    }

    pub fn format_hex(rgb: [f32; 3]) -> String {
        let r = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        format!("#{:02X}{:02X}{:02X}", r, g, b)
    }

    /// Evaluate triplanar procedural surface texture (normal bump, caustic light web, foam, or grid).
    #[inline]
    pub fn evaluate_texture(&self, world_pos: Vec3, base_normal: Vec3, speed: f32, time: f32) -> SurfaceTexSample {
        let strength = self.texture_strength.clamp(0.0, 1.0);
        if self.texture_mode == MeshTextureMode::None || strength <= 1e-4 {
            return SurfaceTexSample {
                normal: base_normal,
                albedo_scale: Vec3::ONE,
                emissive_caustic: 0.0,
                foam_boost: 0.0,
            };
        }

        let scale = self.texture_scale.max(0.2);
        let p = world_pos * scale;

        match self.texture_mode {
            MeshTextureMode::None => SurfaceTexSample {
                normal: base_normal,
                albedo_scale: Vec3::ONE,
                emissive_caustic: 0.0,
                foam_boost: 0.0,
            },
            MeshTextureMode::Ripples => {
                // Multi-directional capillary wave interference (analytical gradient for crisp normal perturbation)
                let t = time * 3.2;
                let k1 = Vec3::new(3.1, 0.7, 2.4);
                let k2 = Vec3::new(-2.3, 1.1, 3.3);
                let k3 = Vec3::new(1.5, -2.9, -2.1);

                let ph1 = p.dot(k1) - t * 1.3;
                let ph2 = p.dot(k2) + t * 1.7;
                let ph3 = p.dot(k3 * 1.6) - t * 2.3;

                let grad = k1 * ph1.cos() * 0.45 + k2 * ph2.cos() * 0.35 + k3 * ph3.cos() * 0.20;
                // Project gradient onto tangent plane so normal only tilts along the surface
                let tangent_grad = grad - base_normal * grad.dot(base_normal);
                let n = (base_normal + tangent_grad * (strength * 0.22)).normalize_or_zero();
                let shimmer = 1.0 + (ph1.sin() * ph2.sin()) * strength * 0.08;
                SurfaceTexSample {
                    normal: if n.length_squared() > 0.5 { n } else { base_normal },
                    albedo_scale: Vec3::splat(shimmer),
                    emissive_caustic: 0.0,
                    foam_boost: 0.0,
                }
            }
            MeshTextureMode::Caustics => {
                // 3D Cellular / Voronoi caustic interference web
                let q = p * 0.85 + Vec3::new(time * 0.55, -time * 0.35, time * 0.45);
                let c1 = voronoi_ridge_3d(q);
                let c2 = voronoi_ridge_3d(q * 1.7 + Vec3::splat(3.7));
                let caustic = (c1 * 0.65 + c2 * 0.35).powf(2.2) * strength;

                // Subtle normal ripple matching caustic cells
                let eps = 0.06;
                let cx = voronoi_ridge_3d(q + Vec3::X * eps) - c1;
                let cz = voronoi_ridge_3d(q + Vec3::Z * eps) - c1;
                let bump = Vec3::new(cx, 0.0, cz) * (strength * 1.8);
                let tangent_bump = bump - base_normal * bump.dot(base_normal);
                let n = (base_normal + tangent_bump).normalize_or_zero();

                SurfaceTexSample {
                    normal: if n.length_squared() > 0.5 { n } else { base_normal },
                    albedo_scale: Vec3::ONE + Vec3::new(0.25, 0.45, 0.55) * caustic,
                    emissive_caustic: caustic * 0.55,
                    foam_boost: 0.0,
                }
            }
            MeshTextureMode::Foam => {
                // Multi-octave FBM organic foam streaks amplified by velocity
                let q = p * 0.9 + Vec3::new(time * 0.3, -time * 0.2, time * 0.25);
                let f = fbm_3d(q);
                let speed_factor = (speed / 2.2).clamp(0.15, 1.0);
                let streaks = ((f - 0.42) * 2.4).clamp(0.0, 1.0) * strength * speed_factor;

                let bump = Vec3::new(f - 0.5, 0.0, fbm_3d(q + Vec3::Z * 1.3) - 0.5) * (strength * 0.30);
                let tangent_bump = bump - base_normal * bump.dot(base_normal);
                let n = (base_normal + tangent_bump).normalize_or_zero();

                SurfaceTexSample {
                    normal: if n.length_squared() > 0.5 { n } else { base_normal },
                    albedo_scale: Vec3::ONE,
                    emissive_caustic: 0.0,
                    foam_boost: streaks * 0.85,
                }
            }
            MeshTextureMode::Grid => {
                // Clean triplanar contour grid blended by surface normal weights
                let w = base_normal.abs().powf(4.0);
                let w_sum = (w.x + w.y + w.z).max(1e-5);
                let weights = w / w_sum;

                let gx = grid_line_2d(Vec2::new(p.y, p.z));
                let gy = grid_line_2d(Vec2::new(p.x, p.z));
                let gz = grid_line_2d(Vec2::new(p.x, p.y));
                let line = gx * weights.x + gy * weights.y + gz * weights.z;

                let checker = (((p.x.floor() as i32) + (p.y.floor() as i32) + (p.z.floor() as i32)) & 1) as f32;
                let base_mod = 1.0 - checker * 0.14 * strength;
                let line_boost = line * strength * 0.65;

                SurfaceTexSample {
                    normal: base_normal,
                    albedo_scale: Vec3::splat(base_mod) + Vec3::splat(line_boost),
                    emissive_caustic: line_boost * 0.25,
                    foam_boost: line_boost * 0.35,
                }
            }
        }
    }
}

#[inline]
fn hash3(p: glam::IVec3) -> Vec3 {
    let mut x = (p.x as u32).wrapping_mul(1597334677)
        ^ (p.y as u32).wrapping_mul(3812015801)
        ^ (p.z as u32).wrapping_mul(2798796415);
    x = x.wrapping_mul(0x27d4eb2d);
    let r1 = ((x & 0xFFFF) as f32) / 65535.0;
    x = x.rotate_right(11).wrapping_mul(0x165667b1);
    let r2 = ((x & 0xFFFF) as f32) / 65535.0;
    x = x.rotate_right(13).wrapping_mul(0x27d4eb2d);
    let r3 = ((x & 0xFFFF) as f32) / 65535.0;
    Vec3::new(r1, r2, r3)
}

#[inline]
fn value_noise_3d(p: Vec3) -> f32 {
    let i = p.floor().as_ivec3();
    let f = p - p.floor();
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);

    let sample = |dx: i32, dy: i32, dz: i32| -> f32 {
        hash3(i + glam::IVec3::new(dx, dy, dz)).x
    };

    let c000 = sample(0, 0, 0);
    let c100 = sample(1, 0, 0);
    let c010 = sample(0, 1, 0);
    let c110 = sample(1, 1, 0);
    let c001 = sample(0, 0, 1);
    let c101 = sample(1, 0, 1);
    let c011 = sample(0, 1, 1);
    let c111 = sample(1, 1, 1);

    let x00 = c000 + (c100 - c000) * u.x;
    let x10 = c010 + (c110 - c010) * u.x;
    let x01 = c001 + (c101 - c001) * u.x;
    let x11 = c011 + (c111 - c011) * u.x;

    let y0 = x00 + (x10 - x00) * u.y;
    let y1 = x01 + (x11 - x01) * u.y;
    y0 + (y1 - y0) * u.z
}

#[inline]
fn fbm_3d(mut p: Vec3) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    for _ in 0..4 {
        sum += amp * value_noise_3d(p);
        p *= 2.03;
        amp *= 0.5;
    }
    sum
}

#[inline]
fn voronoi_ridge_3d(p: Vec3) -> f32 {
    let base = p.floor().as_ivec3();
    let frac = p - p.floor();
    let mut d1 = 8.0f32;
    let mut d2 = 8.0f32;

    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let offset = glam::IVec3::new(dx, dy, dz);
                let jitter = hash3(base + offset);
                let r = offset.as_vec3() + jitter - frac;
                let d = r.length_squared();
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                } else if d < d2 {
                    d2 = d;
                }
            }
        }
    }
    let ridge = 1.0 - ((d2.sqrt() - d1.sqrt()) * 2.2).clamp(0.0, 1.0);
    ridge
}

#[inline]
fn grid_line_2d(uv: Vec2) -> f32 {
    let f = (uv - uv.floor() - Vec2::splat(0.5)).abs();
    let dist_to_edge = (0.5 - f.x).min(0.5 - f.y);
    1.0 - (dist_to_edge / 0.045).clamp(0.0, 1.0)
}

/// GPU/CPU vertex for reconstructed fluid surface triangles.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub speed: f32,
    pub foam: f32,
}

/// Reconstructed 3D triangle mesh of the liquid surface.
#[derive(Clone, Debug, Default)]
pub struct SurfaceMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

const BRICK_DIM: usize = 8;

#[derive(Clone, Copy)]
struct RawEdgeVertex {
    key: u64,
    pos: Vec3,
    vel: Vec3,
}

impl SurfaceMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.vertices.is_empty()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Reconstruct a smooth, watertight 3D liquid surface from SPH fluid particles
    /// using parallel narrow-band brick Marching Cubes + volume-preserving Taubin smoothing.
    pub fn from_particles(
        particles: &[Particle],
        particle_radius: f32,
        smoothing_length: f32,
        config: &SurfaceMeshConfig,
    ) -> Self {
        // Collect fluid particles
        let fluid_positions: Vec<Vec3> = particles
            .iter()
            .filter(|p| {
                p.particle_type == ParticleType::Fluid
                    && p.position.is_finite()
                    && p.velocity.is_finite()
            })
            .map(|p| p.position)
            .collect();
        if fluid_positions.is_empty() {
            return Self::default();
        }
        let fluid_velocities: Vec<Vec3> = particles
            .iter()
            .filter(|p| {
                p.particle_type == ParticleType::Fluid
                    && p.position.is_finite()
                    && p.velocity.is_finite()
            })
            .map(|p| p.velocity)
            .collect();

        let kernel_scale = config.kernel_scale.clamp(0.6, 2.0);
        let splat_radius = (particle_radius * 3.05)
            .max(smoothing_length * 1.32)
            * kernel_scale;
        let splat_r2 = splat_radius * splat_radius;
        let inv_splat_r2 = 1.0 / splat_r2.max(1e-10);

        let min_voxel = config.min_voxel_size.max(0.00025);
        let mut voxel_size = (particle_radius * config.voxel_scale.clamp(0.30, 1.8)).max(min_voxel);

        // Tight bounding box around fluid particles
        let mut min_p = Vec3::splat(f32::INFINITY);
        let mut max_p = Vec3::splat(f32::NEG_INFINITY);
        for &pos in &fluid_positions {
            min_p = min_p.min(pos);
            max_p = max_p.max(pos);
        }

        let pad = splat_radius + voxel_size * 2.0;
        let grid_min = min_p - Vec3::splat(pad);
        let grid_max = max_p + Vec3::splat(pad);
        let extent = grid_max - grid_min;

        // Cap maximum grid resolution along longest axis so memory and time stay bounded
        let max_axis_extent = extent.x.max(extent.y).max(extent.z);
        let max_cells_axis = config.max_grid_cells.max(32.0);
        if max_axis_extent / voxel_size > max_cells_axis {
            voxel_size = max_axis_extent / max_cells_axis;
        }

        let nx = ((extent.x / voxel_size).ceil() as usize).max(2);
        let ny = ((extent.y / voxel_size).ceil() as usize).max(2);
        let nz = ((extent.z / voxel_size).ceil() as usize).max(2);

        let bricks_x = (nx + BRICK_DIM - 1) / BRICK_DIM;
        let bricks_y = (ny + BRICK_DIM - 1) / BRICK_DIM;
        let bricks_z = (nz + BRICK_DIM - 1) / BRICK_DIM;
        let total_bricks = bricks_x * bricks_y * bricks_z;

        // Mark narrow-band active bricks overlapped by at least one particle's splat AABB
        let mut brick_active = vec![false; total_bricks];
        let brick_world_size = voxel_size * BRICK_DIM as f32;
        let inv_brick_size = 1.0 / brick_world_size;

        for &pos in &fluid_positions {
            let b_min = ((pos - Vec3::splat(splat_radius) - grid_min) * inv_brick_size)
                .floor()
                .as_ivec3();
            let b_max = ((pos + Vec3::splat(splat_radius) - grid_min) * inv_brick_size)
                .floor()
                .as_ivec3();

            let z0 = b_min.z.clamp(0, bricks_z as i32 - 1) as usize;
            let z1 = b_max.z.clamp(0, bricks_z as i32 - 1) as usize;
            let y0 = b_min.y.clamp(0, bricks_y as i32 - 1) as usize;
            let y1 = b_max.y.clamp(0, bricks_y as i32 - 1) as usize;
            let x0 = b_min.x.clamp(0, bricks_x as i32 - 1) as usize;
            let x1 = b_max.x.clamp(0, bricks_x as i32 - 1) as usize;

            for bz in z0..=z1 {
                for by in y0..=y1 {
                    let row = (bz * bricks_y + by) * bricks_x;
                    for bx in x0..=x1 {
                        brick_active[row + bx] = true;
                    }
                }
            }
        }

        let active_brick_coords: Vec<[usize; 3]> = brick_active
            .iter()
            .enumerate()
            .filter_map(|(idx, &active)| {
                if !active {
                    return None;
                }
                let bx = idx % bricks_x;
                let rem = idx / bricks_x;
                let by = rem % bricks_y;
                let bz = rem / bricks_y;
                Some([bx, by, bz])
            })
            .collect();

        if active_brick_coords.is_empty() {
            return Self::default();
        }

        // Build spatial hash on fluid positions with cell_size = splat_radius
        let mut p_grid = SpatialHashGrid::new(splat_radius, fluid_positions.len());
        p_grid.build(&fluid_positions);

        let iso = config.iso_value.clamp(0.08, 0.92);

        // Process each active 8x8x8 brick in parallel on all CPU cores
        let brick_outputs: Vec<Vec<[RawEdgeVertex; 3]>> = active_brick_coords
            .par_iter()
            .map(|&[bx, by, bz]| {
                let gx0 = bx * BRICK_DIM;
                let gy0 = by * BRICK_DIM;
                let gz0 = bz * BRICK_DIM;

                let cells_x = BRICK_DIM.min(nx.saturating_sub(gx0));
                let cells_y = BRICK_DIM.min(ny.saturating_sub(gy0));
                let cells_z = BRICK_DIM.min(nz.saturating_sub(gz0));

                // Evaluate 9x9x9 corner scalar values and velocity averages in L1 cache
                const STRIDE_Y: usize = BRICK_DIM + 1; // 9
                const STRIDE_Z: usize = STRIDE_Y * STRIDE_Y; // 81
                let mut field = [0.0f32; (BRICK_DIM + 1) * (BRICK_DIM + 1) * (BRICK_DIM + 1)];
                let mut vels = [Vec3::ZERO; (BRICK_DIM + 1) * (BRICK_DIM + 1) * (BRICK_DIM + 1)];

                let mut any_inside = false;
                let mut any_outside = false;

                for lz in 0..=cells_z {
                    let wz = grid_min.z + (gz0 + lz) as f32 * voxel_size;
                    for ly in 0..=cells_y {
                        let wy = grid_min.y + (gy0 + ly) as f32 * voxel_size;
                        for lx in 0..=cells_x {
                            let wx = grid_min.x + (gx0 + lx) as f32 * voxel_size;
                            let sample_pos = Vec3::new(wx, wy, wz);

                            let mut sum_val = 0.0f32;
                            let mut weighted_vel = Vec3::ZERO;

                            p_grid.for_each_neighbor(sample_pos, |j| {
                                let d2 = sample_pos.distance_squared(fluid_positions[j]);
                                if d2 < splat_r2 {
                                    let t = 1.0 - d2 * inv_splat_r2;
                                    let w = t * t * t; // Smooth C2 polynomial kernel
                                    sum_val += w;
                                    weighted_vel += fluid_velocities[j] * w;
                                }
                            });

                            // Soft saturation prevents extreme density spikes in deep pools while keeping iso=0.42 exact
                            let phi = sum_val / (1.0 + sum_val * 0.35);
                            let idx = lz * STRIDE_Z + ly * STRIDE_Y + lx;
                            field[idx] = phi;
                            if sum_val > 1e-6 {
                                vels[idx] = weighted_vel / sum_val;
                            }

                            if phi >= iso {
                                any_inside = true;
                            } else {
                                any_outside = true;
                            }
                        }
                    }
                }

                let mut tris = Vec::new();
                if !any_inside || !any_outside {
                    return tris;
                }

                // Extract Marching Cubes triangles within this brick
                for lz in 0..cells_z {
                    for ly in 0..cells_y {
                        for lx in 0..cells_x {
                            let c0 = lz * STRIDE_Z + ly * STRIDE_Y + lx;
                            let c1 = c0 + 1;
                            let c2 = c0 + STRIDE_Y + 1;
                            let c3 = c0 + STRIDE_Y;
                            let c4 = c0 + STRIDE_Z;
                            let c5 = c4 + 1;
                            let c6 = c4 + STRIDE_Y + 1;
                            let c7 = c4 + STRIDE_Z - STRIDE_Z + STRIDE_Y; // c4 + STRIDE_Y

                            let v = [
                                field[c0], field[c1], field[c2], field[c3],
                                field[c4], field[c5], field[c6], field[c7],
                            ];

                            let mut cube_index = 0usize;
                            if v[0] >= iso { cube_index |= 1; }
                            if v[1] >= iso { cube_index |= 2; }
                            if v[2] >= iso { cube_index |= 4; }
                            if v[3] >= iso { cube_index |= 8; }
                            if v[4] >= iso { cube_index |= 16; }
                            if v[5] >= iso { cube_index |= 32; }
                            if v[6] >= iso { cube_index |= 64; }
                            if v[7] >= iso { cube_index |= 128; }

                            let edge_mask = MC_EDGE_TABLE[cube_index];
                            if edge_mask == 0 {
                                continue;
                            }

                            let gx = gx0 + lx;
                            let gy = gy0 + ly;
                            let gz = gz0 + lz;
                            let p0 = grid_min + Vec3::new(gx as f32, gy as f32, gz as f32) * voxel_size;

                            let corner_pos = [
                                p0,
                                p0 + Vec3::new(voxel_size, 0.0, 0.0),
                                p0 + Vec3::new(voxel_size, voxel_size, 0.0),
                                p0 + Vec3::new(0.0, voxel_size, 0.0),
                                p0 + Vec3::new(0.0, 0.0, voxel_size),
                                p0 + Vec3::new(voxel_size, 0.0, voxel_size),
                                p0 + Vec3::new(voxel_size, voxel_size, voxel_size),
                                p0 + Vec3::new(0.0, voxel_size, voxel_size),
                            ];
                            let corner_vel = [
                                vels[c0], vels[c1], vels[c2], vels[c3],
                                vels[c4], vels[c5], vels[c6], vels[c7],
                            ];

                            let mut edge_verts = [RawEdgeVertex {
                                key: 0,
                                pos: Vec3::ZERO,
                                vel: Vec3::ZERO,
                            }; 12];

                            for e in 0..12 {
                                if (edge_mask & (1 << e)) != 0 {
                                    let (a, b) = MC_EDGE_CORNERS[e];
                                    let va = v[a];
                                    let vb = v[b];
                                    let denom = vb - va;
                                    let t = if denom.abs() > 1e-6 {
                                        ((iso - va) / denom).clamp(0.0, 1.0)
                                    } else {
                                        0.5
                                    };
                                    let pos = corner_pos[a].lerp(corner_pos[b], t);
                                    let vel = corner_vel[a].lerp(corner_vel[b], t);
                                    let key = pack_grid_edge_key(gx, gy, gz, e);
                                    edge_verts[e] = RawEdgeVertex { key, pos, vel };
                                }
                            }

                            let row = &MC_TRI_TABLE[cube_index];
                            let mut t_idx = 0;
                            while t_idx < 15 && row[t_idx] != -1 {
                                let e0 = row[t_idx] as usize;
                                let e1 = row[t_idx + 1] as usize;
                                let e2 = row[t_idx + 2] as usize;
                                // Winding (e0, e2, e1) points outward along -grad(phi) in right-handed coordinates
                                tris.push([edge_verts[e0], edge_verts[e2], edge_verts[e1]]);
                                t_idx += 3;
                            }
                        }
                    }
                }

                tris
            })
            .collect();

        // Weld shared edge vertices into an indexed manifold mesh
        let total_tris: usize = brick_outputs.iter().map(|v| v.len()).sum();
        if total_tris == 0 {
            return Self::default();
        }

        let mut key_to_idx: HashMap<u64, u32> = HashMap::with_capacity(total_tris);
        let mut positions: Vec<Vec3> = Vec::with_capacity(total_tris / 2 + 16);
        let mut velocities: Vec<Vec3> = Vec::with_capacity(total_tris / 2 + 16);
        let mut indices: Vec<u32> = Vec::with_capacity(total_tris * 3);

        for brick_tris in &brick_outputs {
            for tri in brick_tris {
                let mut tri_idx = [0u32; 3];
                for k in 0..3 {
                    let ev = tri[k];
                    let v_id = *key_to_idx.entry(ev.key).or_insert_with(|| {
                        let id = positions.len() as u32;
                        positions.push(ev.pos);
                        velocities.push(ev.vel);
                        id
                    });
                    tri_idx[k] = v_id;
                }
                if tri_idx[0] != tri_idx[1] && tri_idx[1] != tri_idx[2] && tri_idx[2] != tri_idx[0] {
                    indices.push(tri_idx[0]);
                    indices.push(tri_idx[1]);
                    indices.push(tri_idx[2]);
                }
            }
        }

        let v_count = positions.len();
        if v_count == 0 || indices.is_empty() {
            return Self::default();
        }

        // Volume-preserving Taubin (lambda / mu) mesh smoothing to erase particle blobbiness
        let smooth_iters = config
            .smoothing_passes
            .min(config.max_smoothing_passes.max(1));
        if smooth_iters > 0 {
            let mut sum_pos = vec![Vec3::ZERO; v_count];
            let mut sum_vel = vec![Vec3::ZERO; v_count];
            let mut valence = vec![0u32; v_count];

            let taubin_step = |positions: &mut [Vec3],
                               velocities: &mut [Vec3],
                               sum_pos: &mut [Vec3],
                               sum_vel: &mut [Vec3],
                               valence: &mut [u32],
                               factor: f32,
                               smooth_vel: bool| {
                sum_pos.fill(Vec3::ZERO);
                if smooth_vel {
                    sum_vel.fill(Vec3::ZERO);
                }
                valence.fill(0);

                for tri in indices.chunks_exact(3) {
                    let i0 = tri[0] as usize;
                    let i1 = tri[1] as usize;
                    let i2 = tri[2] as usize;

                    let p0 = positions[i0];
                    let p1 = positions[i1];
                    let p2 = positions[i2];

                    sum_pos[i0] += p1 + p2;
                    sum_pos[i1] += p2 + p0;
                    sum_pos[i2] += p0 + p1;

                    valence[i0] += 2;
                    valence[i1] += 2;
                    valence[i2] += 2;

                    if smooth_vel {
                        let v0 = velocities[i0];
                        let v1 = velocities[i1];
                        let v2 = velocities[i2];
                        sum_vel[i0] += v1 + v2;
                        sum_vel[i1] += v2 + v0;
                        sum_vel[i2] += v0 + v1;
                    }
                }

                for i in 0..positions.len() {
                    let deg = valence[i];
                    if deg > 0 {
                        let inv_deg = 1.0 / deg as f32;
                        let avg_p = sum_pos[i] * inv_deg;
                        positions[i] += (avg_p - positions[i]) * factor;
                        if smooth_vel {
                            let avg_v = sum_vel[i] * inv_deg;
                            velocities[i] = velocities[i].lerp(avg_v, 0.35);
                        }
                    }
                }
            };

            let lambda = 0.50f32;
            let mu = -0.522f32;
            for _ in 0..smooth_iters {
                taubin_step(&mut positions, &mut velocities, &mut sum_pos, &mut sum_vel, &mut valence, lambda, true);
                taubin_step(&mut positions, &mut velocities, &mut sum_pos, &mut sum_vel, &mut valence, mu, false);
            }
            // Final micro-polish pass eliminates sub-voxel Marching Cubes faceting
            taubin_step(&mut positions, &mut velocities, &mut sum_pos, &mut sum_vel, &mut valence, 0.24, false);
        }

        // Compute area-weighted smooth vertex normals + curvature crest foam cue
        let mut normals = vec![Vec3::ZERO; v_count];
        for tri in indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            let e1 = positions[i1] - positions[i0];
            let e2 = positions[i2] - positions[i0];
            let fn_area = e1.cross(e2);
            normals[i0] += fn_area;
            normals[i1] += fn_area;
            normals[i2] += fn_area;
        }

        let vertices: Vec<MeshVertex> = (0..v_count)
            .into_par_iter()
            .map(|i| {
                let n = if normals[i].length_squared() > 1e-16 {
                    normals[i].normalize()
                } else {
                    Vec3::Y
                };
                let speed = velocities[i].length();
                // Crest foam combines speed and upward/splash motion
                let upward_splash = (velocities[i].y * 0.25).clamp(0.0, 0.4);
                let foam = ((speed - 1.35) / 2.6 + upward_splash).clamp(0.0, 1.0);
                MeshVertex {
                    position: positions[i].to_array(),
                    normal: n.to_array(),
                    speed,
                    foam,
                }
            })
            .collect();

        Self { vertices, indices }
    }

    /// Export reconstructed surface mesh to Wavefront `.obj` file (with vertex positions and smooth normals).
    pub fn save_obj(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut out = BufWriter::new(file);
        writeln!(out, "# LiquidRust Reconstructed Fluid Surface Mesh")?;
        writeln!(
            out,
            "# Vertices: {}, Triangles: {}",
            self.vertices.len(),
            self.triangle_count()
        )?;

        for v in &self.vertices {
            writeln!(
                out,
                "v {:.6} {:.6} {:.6}",
                v.position[0], v.position[1], v.position[2]
            )?;
        }
        for v in &self.vertices {
            writeln!(
                out,
                "vn {:.6} {:.6} {:.6}",
                v.normal[0], v.normal[1], v.normal[2]
            )?;
        }
        for tri in self.indices.chunks_exact(3) {
            let a = tri[0] + 1;
            let b = tri[1] + 1;
            let c = tri[2] + 1;
            writeln!(out, "f {}//{} {}//{} {}//{}", a, a, b, b, c, c)?;
        }
        out.flush()
    }
}

#[inline(always)]
fn pack_grid_edge_key(gx: usize, gy: usize, gz: usize, edge_idx: usize) -> u64 {
    let (ox, oy, oz, axis) = match edge_idx {
        0 => (0, 0, 0, 0u64),
        1 => (1, 0, 0, 1u64),
        2 => (0, 1, 0, 0u64),
        3 => (0, 0, 0, 1u64),
        4 => (0, 0, 1, 0u64),
        5 => (1, 0, 1, 1u64),
        6 => (0, 1, 1, 0u64),
        7 => (0, 0, 1, 1u64),
        8 => (0, 0, 0, 2u64),
        9 => (1, 0, 0, 2u64),
        10 => (1, 1, 0, 2u64),
        11 => (0, 1, 0, 2u64),
        _ => (0, 0, 0, 0u64),
    };
    let x = (gx + ox) as u64;
    let y = (gy + oy) as u64;
    let z = (gz + oz) as u64;
    (x << 42) | (y << 22) | (z << 2) | axis
}

pub const MC_EDGE_CORNERS: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

pub const MC_EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

pub const MC_TRI_TABLE: [[i8; 16]; 256] = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
];
