use crate::math::Aabb;
use crate::sph::particle::Particle;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMapType {
    Ocean,
    Turbo,
    Viridis,
    Fire,
    Plasma,
    Inferno,
    Ice,
}

impl ColorMapType {
    pub const ALL: &'static [ColorMapType] = &[
        ColorMapType::Ocean,
        ColorMapType::Turbo,
        ColorMapType::Viridis,
        ColorMapType::Fire,
        ColorMapType::Plasma,
        ColorMapType::Inferno,
        ColorMapType::Ice,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ColorMapType::Ocean => "Ocean (Aqua -> White)",
            ColorMapType::Turbo => "Turbo (Rainbow)",
            ColorMapType::Viridis => "Viridis (Perceptual)",
            ColorMapType::Fire => "Fire (Thermal)",
            ColorMapType::Plasma => "Plasma (Magenta -> Gold)",
            ColorMapType::Inferno => "Inferno (Magma)",
            ColorMapType::Ice => "Arctic Ice (Glacier)",
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            ColorMapType::Ocean => "ocean",
            ColorMapType::Turbo => "turbo",
            ColorMapType::Viridis => "viridis",
            ColorMapType::Fire => "fire",
            ColorMapType::Plasma => "plasma",
            ColorMapType::Inferno => "inferno",
            ColorMapType::Ice => "ice",
        }
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ocean" | "water" => Some(ColorMapType::Ocean),
            "turbo" | "rainbow" => Some(ColorMapType::Turbo),
            "viridis" => Some(ColorMapType::Viridis),
            "fire" | "thermal" => Some(ColorMapType::Fire),
            "plasma" => Some(ColorMapType::Plasma),
            "inferno" | "magma" | "lava" => Some(ColorMapType::Inferno),
            "ice" | "arctic" | "glacier" => Some(ColorMapType::Ice),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Velocity,
    Pressure,
    Density,
    Height,
    KineticEnergy,
    Foam,
    Vorticity,
    SingleColor,
}

impl ColorMode {
    pub const ALL: &'static [ColorMode] = &[
        ColorMode::Velocity,
        ColorMode::Pressure,
        ColorMode::Density,
        ColorMode::Height,
        ColorMode::KineticEnergy,
        ColorMode::Foam,
        ColorMode::Vorticity,
        ColorMode::SingleColor,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ColorMode::Velocity => "Velocity Magnitude",
            ColorMode::Pressure => "Pressure Field",
            ColorMode::Density => "Density Deviation",
            ColorMode::Height => "Height (Y)",
            ColorMode::KineticEnergy => "Kinetic Energy",
            ColorMode::Foam => "Whitewater Aeration",
            ColorMode::Vorticity => "Swirl / Vertical Shear",
            ColorMode::SingleColor => "Uniform Aqua",
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            ColorMode::Velocity => "velocity",
            ColorMode::Pressure => "pressure",
            ColorMode::Density => "density",
            ColorMode::Height => "height",
            ColorMode::KineticEnergy => "energy",
            ColorMode::Foam => "foam",
            ColorMode::Vorticity => "vorticity",
            ColorMode::SingleColor => "single",
        }
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "velocity" | "vel" | "speed" => Some(ColorMode::Velocity),
            "pressure" | "press" => Some(ColorMode::Pressure),
            "density" | "dens" => Some(ColorMode::Density),
            "height" | "y" => Some(ColorMode::Height),
            "energy" | "ke" | "kinetic" => Some(ColorMode::KineticEnergy),
            "foam" | "aeration" | "spray" => Some(ColorMode::Foam),
            "vorticity" | "swirl" | "curl" => Some(ColorMode::Vorticity),
            "single" | "uniform" | "aqua" => Some(ColorMode::SingleColor),
            _ => None,
        }
    }
}

pub struct ColorMap;

impl ColorMap {
    /// Sample color given normalized value t in [0.0, 1.0]
    pub fn sample(cmap: ColorMapType, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let rgb = match cmap {
            ColorMapType::Ocean => Self::sample_ocean(t),
            ColorMapType::Turbo => Self::sample_turbo(t),
            ColorMapType::Viridis => Self::sample_viridis(t),
            ColorMapType::Fire => Self::sample_fire(t),
            ColorMapType::Plasma => Self::sample_plasma(t),
            ColorMapType::Inferno => Self::sample_inferno(t),
            ColorMapType::Ice => Self::sample_ice(t),
        };
        [
            (rgb.x * 255.0).clamp(0.0, 255.0) as u8,
            (rgb.y * 255.0).clamp(0.0, 255.0) as u8,
            (rgb.z * 255.0).clamp(0.0, 255.0) as u8,
        ]
    }

    /// Evaluate per-particle color with optional aerated whitewater crest highlights
    pub fn particle_color(
        p: &Particle,
        mode: ColorMode,
        cmap: ColorMapType,
        rest_density: f32,
        bounds: &Aabb,
        particle_foam: bool,
    ) -> [u8; 3] {
        let rho0 = rest_density.max(1e-4);
        let speed = p.speed();
        let mut rgb = match mode {
            ColorMode::Velocity => Self::sample(cmap, (speed / 2.6).clamp(0.0, 1.0)),
            ColorMode::Pressure => Self::sample(cmap, (p.pressure / (rho0 * 0.4)).clamp(0.0, 1.0)),
            ColorMode::Density => {
                let dev = (p.density - rho0).abs() / rho0;
                Self::sample(cmap, (dev * 4.0).clamp(0.0, 1.0))
            }
            ColorMode::Height => {
                let h = (bounds.max.y - bounds.min.y).max(1e-4);
                Self::sample(cmap, ((p.position.y - bounds.min.y) / h).clamp(0.0, 1.0))
            }
            ColorMode::KineticEnergy => {
                Self::sample(cmap, (p.kinetic_energy() / 4.0).clamp(0.0, 1.0))
            }
            ColorMode::Foam => {
                let rho_def = (1.0 - p.density / rho0).clamp(0.0, 1.0);
                let aeration = (rho_def * 1.45 * (speed / 2.2).clamp(0.15, 1.2)).clamp(0.0, 1.0);
                Self::sample(cmap, aeration)
            }
            ColorMode::Vorticity => {
                let horiz = (p.velocity.x * p.velocity.x + p.velocity.z * p.velocity.z).sqrt();
                let shear = ((horiz * 0.65 + p.velocity.y.abs() * 0.85) / 2.8).clamp(0.0, 1.0);
                Self::sample(cmap, shear)
            }
            ColorMode::SingleColor => [30, 140, 225],
        };

        if particle_foam && mode != ColorMode::SingleColor {
            let rho_ratio = p.density / rho0;
            if speed > 1.15 && rho_ratio < 0.78 {
                let spd_f = ((speed - 1.15) / 2.6).clamp(0.0, 1.0);
                let air_f = ((0.78 - rho_ratio) / 0.36).clamp(0.0, 1.0);
                let foam = (spd_f * air_f * 0.85).clamp(0.0, 0.85);
                rgb[0] = (rgb[0] as f32 * (1.0 - foam) + 244.0 * foam) as u8;
                rgb[1] = (rgb[1] as f32 * (1.0 - foam) + 250.0 * foam) as u8;
                rgb[2] = (rgb[2] as f32 * (1.0 - foam) + 255.0 * foam) as u8;
            }
        }
        rgb
    }

    fn sample_ocean(t: f32) -> Vec3 {
        // Deep blue (rest) -> cyan/turquoise (flow) -> white (splash)
        if t < 0.25 {
            let s = t / 0.25;
            Vec3::new(0.05, 0.22, 0.65).lerp(Vec3::new(0.08, 0.45, 0.85), s)
        } else if t < 0.65 {
            let s = (t - 0.25) / 0.40;
            Vec3::new(0.08, 0.45, 0.85).lerp(Vec3::new(0.18, 0.80, 0.94), s)
        } else {
            let s = (t - 0.65) / 0.35;
            Vec3::new(0.18, 0.80, 0.94).lerp(Vec3::new(0.96, 0.98, 1.0), s)
        }
    }

    fn sample_turbo(t: f32) -> Vec3 {
        // Approximate Google Turbo colormap polynomial
        let r = 0.1357 + t * (4.5974 - t * (42.3277 - t * (130.5887 - t * (150.5667 - t * 58.1375))));
        let g = 0.0914 + t * (2.1856 + t * (4.8052 - t * (14.0195 - t * (4.2109 - t * 2.7747))));
        let b = 0.1067 + t * (12.5732 - t * (80.1472 - t * (175.8194 - t * (154.5829 - t * 47.7818))));
        Vec3::new(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
    }

    fn sample_viridis(t: f32) -> Vec3 {
        if t < 0.25 {
            let s = t / 0.25;
            Vec3::new(0.267, 0.004, 0.329).lerp(Vec3::new(0.231, 0.322, 0.545), s)
        } else if t < 0.5 {
            let s = (t - 0.25) / 0.25;
            Vec3::new(0.231, 0.322, 0.545).lerp(Vec3::new(0.129, 0.569, 0.553), s)
        } else if t < 0.75 {
            let s = (t - 0.5) / 0.25;
            Vec3::new(0.129, 0.569, 0.553).lerp(Vec3::new(0.369, 0.788, 0.384), s)
        } else {
            let s = (t - 0.75) / 0.25;
            Vec3::new(0.369, 0.788, 0.384).lerp(Vec3::new(0.993, 0.906, 0.144), s)
        }
    }

    fn sample_fire(t: f32) -> Vec3 {
        if t < 0.33 {
            let s = t / 0.33;
            Vec3::new(0.05, 0.02, 0.05).lerp(Vec3::new(0.7, 0.1, 0.05), s)
        } else if t < 0.66 {
            let s = (t - 0.33) / 0.33;
            Vec3::new(0.7, 0.1, 0.05).lerp(Vec3::new(0.95, 0.6, 0.1), s)
        } else {
            let s = (t - 0.66) / 0.34;
            Vec3::new(0.95, 0.6, 0.1).lerp(Vec3::new(1.0, 1.0, 0.8), s)
        }
    }

    fn sample_plasma(t: f32) -> Vec3 {
        if t < 0.5 {
            let s = t / 0.5;
            Vec3::new(0.05, 0.03, 0.53).lerp(Vec3::new(0.75, 0.22, 0.52), s)
        } else {
            let s = (t - 0.5) / 0.5;
            Vec3::new(0.75, 0.22, 0.52).lerp(Vec3::new(0.94, 0.97, 0.13), s)
        }
    }

    fn sample_inferno(t: f32) -> Vec3 {
        if t < 0.33 {
            let s = t / 0.33;
            Vec3::new(0.01, 0.01, 0.08).lerp(Vec3::new(0.45, 0.08, 0.43), s)
        } else if t < 0.68 {
            let s = (t - 0.33) / 0.35;
            Vec3::new(0.45, 0.08, 0.43).lerp(Vec3::new(0.93, 0.42, 0.14), s)
        } else {
            let s = (t - 0.68) / 0.32;
            Vec3::new(0.93, 0.42, 0.14).lerp(Vec3::new(0.99, 0.99, 0.64), s)
        }
    }

    fn sample_ice(t: f32) -> Vec3 {
        if t < 0.35 {
            let s = t / 0.35;
            Vec3::new(0.03, 0.08, 0.22).lerp(Vec3::new(0.12, 0.42, 0.68), s)
        } else if t < 0.72 {
            let s = (t - 0.35) / 0.37;
            Vec3::new(0.12, 0.42, 0.68).lerp(Vec3::new(0.56, 0.88, 0.96), s)
        } else {
            let s = (t - 0.72) / 0.28;
            Vec3::new(0.56, 0.88, 0.96).lerp(Vec3::new(0.98, 0.995, 1.0), s)
        }
    }
}
