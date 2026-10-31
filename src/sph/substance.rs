#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialType {
    Water,
    Honey,
    Milk,
    Oil,
    Alcohol,
    Gas,
    Fire,
    Steam,
    Smoke,
}

impl MaterialType {
    pub const ALL: &'static [MaterialType] = &[
        MaterialType::Water,
        MaterialType::Honey,
        MaterialType::Milk,
        MaterialType::Oil,
        MaterialType::Alcohol,
        MaterialType::Gas,
        MaterialType::Fire,
        MaterialType::Steam,
        MaterialType::Smoke,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            MaterialType::Water => "💧 Water",
            MaterialType::Honey => "🍯 Honey",
            MaterialType::Milk => "🥛 Milk",
            MaterialType::Oil => "🛢 Oil",
            MaterialType::Alcohol => "🍸 Alcohol",
            MaterialType::Gas => "☁ Gas (Methane)",
            MaterialType::Fire => "🔥 Fire",
            MaterialType::Steam => "♨ Steam",
            MaterialType::Smoke => "💨 Smoke",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().trim() {
            "water" | "aqua" => Some(MaterialType::Water),
            "honey" | "syrup" => Some(MaterialType::Honey),
            "milk" | "dairy" => Some(MaterialType::Milk),
            "oil" | "petrol" => Some(MaterialType::Oil),
            "alcohol" | "ethanol" | "liquor" => Some(MaterialType::Alcohol),
            "gas" | "methane" | "vapor" => Some(MaterialType::Gas),
            "fire" | "flame" | "burn" => Some(MaterialType::Fire),
            "steam" | "boil" => Some(MaterialType::Steam),
            "smoke" => Some(MaterialType::Smoke),
            _ => None,
        }
    }

    /// GPU material code: 0..=8
    pub fn code(&self) -> u32 {
        match self {
            MaterialType::Water => 0,
            MaterialType::Honey => 1,
            MaterialType::Milk => 2,
            MaterialType::Oil => 3,
            MaterialType::Alcohol => 4,
            MaterialType::Gas => 5,
            MaterialType::Fire => 6,
            MaterialType::Steam => 7,
            MaterialType::Smoke => 8,
        }
    }

    pub fn from_code(code: u32) -> Self {
        match code {
            1 => MaterialType::Honey,
            2 => MaterialType::Milk,
            3 => MaterialType::Oil,
            4 => MaterialType::Alcohol,
            5 => MaterialType::Gas,
            6 => MaterialType::Fire,
            7 => MaterialType::Steam,
            8 => MaterialType::Smoke,
            _ => MaterialType::Water,
        }
    }

    /// Dynamic viscosity coefficient for viscous damping
    pub fn viscosity(&self) -> f32 {
        match self {
            MaterialType::Water => 0.05,
            MaterialType::Honey => 0.42,   // thick, slow-oozing, coiling liquid
            MaterialType::Milk => 0.07,
            MaterialType::Oil => 0.20,    // smooth viscous lubricant
            MaterialType::Alcohol => 0.02, // very fluid
            MaterialType::Gas => 0.01,
            MaterialType::Fire => 0.02,
            MaterialType::Steam => 0.015,
            MaterialType::Smoke => 0.02,
        }
    }

    /// Relative density factor (Water = 1.0)
    pub fn density_factor(&self) -> f32 {
        match self {
            MaterialType::Water => 1.0,
            MaterialType::Honey => 1.42,  // sinks beneath water
            MaterialType::Milk => 1.03,
            MaterialType::Oil => 0.91,    // floats on water!
            MaterialType::Alcohol => 0.79,// floats on oil!
            MaterialType::Gas => 0.15,
            MaterialType::Fire => 0.08,
            MaterialType::Steam => 0.20,
            MaterialType::Smoke => 0.25,
        }
    }

    /// Upward buoyancy acceleration in m/s^2 (counters or exceeds gravity)
    pub fn buoyancy(&self) -> f32 {
        match self {
            MaterialType::Gas => 14.5,    // buoyant gas drifts upward
            MaterialType::Fire => 32.0,   // hot combustion plume rises rapidly
            MaterialType::Steam => 22.0,  // boiling steam billows upward
            MaterialType::Smoke => 10.0,  // smoke gently rises
            _ => 0.0,
        }
    }

    /// Surface tension coefficient
    pub fn surface_tension(&self) -> f32 {
        match self {
            MaterialType::Water => 0.08,
            MaterialType::Honey => 0.22,
            MaterialType::Milk => 0.06,
            MaterialType::Oil => 0.16,
            MaterialType::Alcohol => 0.03,
            _ => 0.0,
        }
    }

    /// Restitution / bounciness
    pub fn restitution(&self) -> f32 {
        match self {
            MaterialType::Honey => 0.0,   // zero bounce
            MaterialType::Oil => 0.02,
            MaterialType::Water => 0.05,
            MaterialType::Alcohol => 0.08,
            _ => 0.0,
        }
    }

    /// Initial temperature in Celsius
    pub fn initial_temperature(&self) -> f32 {
        match self {
            MaterialType::Fire => 650.0,
            MaterialType::Steam => 110.0,
            MaterialType::Smoke => 90.0,
            _ => 20.0,
        }
    }

    /// Initial lifespan in seconds (for transient phase particles)
    pub fn initial_life(&self) -> f32 {
        match self {
            MaterialType::Fire => 1.2,
            MaterialType::Steam => 3.5,
            MaterialType::Smoke => 4.0,
            _ => 1.0,
        }
    }

    /// Natural visual color [R, G, B]
    pub fn color(&self, temperature: f32) -> [u8; 3] {
        match self {
            MaterialType::Water => {
                if temperature >= 90.0 {
                    [70, 185, 245] // warm near-boiling cyan
                } else {
                    [28, 140, 230] // crystal azure blue
                }
            }
            MaterialType::Honey => [235, 160, 20],   // rich golden amber
            MaterialType::Milk => [245, 242, 236],   // creamy opaque white
            MaterialType::Oil => [195, 180, 42],     // golden olive yellow
            MaterialType::Alcohol => [125, 180, 255],// pale bluish-violet
            MaterialType::Gas => [65, 225, 180],    // glowing translucent green/cyan
            MaterialType::Fire => {
                // Fiery gradient: yellow core -> fiery orange -> red tips
                let factor = (temperature / 800.0).clamp(0.0, 1.0);
                let g = (100.0 + factor * 155.0) as u8;
                let b = (factor * 60.0) as u8;
                [255, g, b]
            }
            MaterialType::Steam => [225, 232, 242], // soft white-grey billow
            MaterialType::Smoke => [85, 85, 92],    // charcoal grey
        }
    }

    pub fn is_gas(&self) -> bool {
        matches!(self, MaterialType::Gas | MaterialType::Fire | MaterialType::Steam | MaterialType::Smoke)
    }

    pub fn is_flammable(&self) -> bool {
        matches!(self, MaterialType::Gas | MaterialType::Alcohol | MaterialType::Oil)
    }
}
