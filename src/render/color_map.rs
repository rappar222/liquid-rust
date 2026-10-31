use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMapType {
    Ocean,
    Turbo,
    Viridis,
    Fire,
    Plasma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Velocity,
    Pressure,
    Density,
    Height,
    KineticEnergy,
    SingleColor,
}

impl ColorMode {
    pub const ALL: &'static [ColorMode] = &[
        ColorMode::Velocity,
        ColorMode::Pressure,
        ColorMode::Density,
        ColorMode::Height,
        ColorMode::KineticEnergy,
        ColorMode::SingleColor,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ColorMode::Velocity => "Velocity Magnitude",
            ColorMode::Pressure => "Pressure Field",
            ColorMode::Density => "Density Deviation",
            ColorMode::Height => "Height (Y)",
            ColorMode::KineticEnergy => "Kinetic Energy",
            ColorMode::SingleColor => "Uniform Aqua",
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
        };
        [
            (rgb.x * 255.0).clamp(0.0, 255.0) as u8,
            (rgb.y * 255.0).clamp(0.0, 255.0) as u8,
            (rgb.z * 255.0).clamp(0.0, 255.0) as u8,
        ]
    }

    fn sample_ocean(t: f32) -> Vec3 {
        // Deep blue -> cerulean -> bright cyan -> white spray
        if t < 0.33 {
            let s = t / 0.33;
            Vec3::new(0.05, 0.2, 0.6).lerp(Vec3::new(0.1, 0.5, 0.85), s)
        } else if t < 0.75 {
            let s = (t - 0.33) / 0.42;
            Vec3::new(0.1, 0.5, 0.85).lerp(Vec3::new(0.3, 0.85, 0.95), s)
        } else {
            let s = (t - 0.75) / 0.25;
            Vec3::new(0.3, 0.85, 0.95).lerp(Vec3::new(0.95, 0.98, 1.0), s)
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
}
