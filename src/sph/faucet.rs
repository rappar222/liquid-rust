use crate::math::Aabb;
use crate::sph::particle::Particle;
use glam::Vec3;
use std::f32::consts::PI;

/// Shape of fluid emitter nozzle / weir
#[derive(Clone, Debug, PartialEq)]
pub enum FaucetShape {
    /// Circular nozzle (showerhead, tap, pipe)
    Circle { radius: f32 },
    /// Rectangular weir / slot (waterfall sheet)
    Rectangle { width: f32, depth: f32 },
    /// Point emitter (pinhole needle jet)
    Point,
}

impl FaucetShape {
    pub fn name(&self) -> &'static str {
        match self {
            FaucetShape::Circle { .. } => "Circle (Shower / Pipe)",
            FaucetShape::Rectangle { .. } => "Rectangle (Waterfall Sheet)",
            FaucetShape::Point => "Point Jet",
        }
    }
}

/// Continuous fluid inflow source (Faucet, Shower, Waterfall)
#[derive(Clone, Debug)]
pub struct Faucet {
    pub id: u32,
    pub name: String,
    pub shape: FaucetShape,
    pub position: Vec3,
    pub direction: Vec3,      // Flow velocity vector (m/s)
    pub flow_rate: f32,      // Particles per second
    pub spray_spread: f32,   // Spray divergence angle in radians (0.0 = laminar, 0.25 = shower cone)
    pub enabled: bool,
    pub accumulator: f32,
    seed: u32,
}

impl Faucet {
    pub fn new_circle(name: impl Into<String>, position: Vec3, direction: Vec3, radius: f32, flow_rate: f32) -> Self {
        Self {
            id: 0,
            name: name.into(),
            shape: FaucetShape::Circle { radius },
            position,
            direction,
            flow_rate,
            spray_spread: 0.0,
            enabled: true,
            accumulator: 0.0,
            seed: 1234567,
        }
    }

    pub fn new_waterfall(name: impl Into<String>, position: Vec3, width: f32, depth: f32, flow_rate: f32) -> Self {
        Self {
            id: 0,
            name: name.into(),
            shape: FaucetShape::Rectangle { width, depth },
            position,
            direction: Vec3::new(0.0, -2.5, 0.0),
            flow_rate,
            spray_spread: 0.0,
            enabled: true,
            accumulator: 0.0,
            seed: 7654321,
        }
    }

    pub fn new_shower(name: impl Into<String>, position: Vec3, radius: f32, flow_rate: f32, spray_spread: f32) -> Self {
        Self {
            id: 0,
            name: name.into(),
            shape: FaucetShape::Circle { radius },
            position,
            direction: Vec3::new(0.0, -3.0, 0.0),
            flow_rate,
            spray_spread,
            enabled: true,
            accumulator: 0.0,
            seed: 9999999,
        }
    }

    pub fn new_point(name: impl Into<String>, position: Vec3, direction: Vec3, flow_rate: f32) -> Self {
        Self {
            id: 0,
            name: name.into(),
            shape: FaucetShape::Point,
            position,
            direction,
            flow_rate,
            spray_spread: 0.0,
            enabled: true,
            accumulator: 0.0,
            seed: 5555555,
        }
    }

    fn next_random(&mut self) -> f32 {
        // Fast lightweight Xorshift32 PRNG
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        (self.seed as f32) / (u32::MAX as f32)
    }

    /// Emit particles for time step dt
    pub fn emit(&mut self, dt: f32, _particle_spacing: f32, next_id: &mut u32) -> Vec<Particle> {
        if !self.enabled || self.flow_rate <= 0.0 {
            return Vec::new();
        }

        self.accumulator += self.flow_rate * dt;
        let count = self.accumulator.floor() as usize;
        self.accumulator -= count as f32;

        if count == 0 {
            return Vec::new();
        }

        let mut particles = Vec::with_capacity(count);
        let base_vel = self.direction;
        let speed = base_vel.length();
        let flow_dir = if speed > 1e-6 { base_vel / speed } else { -Vec3::Y };

        // Construct orthonormal basis around flow direction
        let up = if flow_dir.y.abs() < 0.95 { Vec3::Y } else { Vec3::X };
        let right = flow_dir.cross(up).normalize();
        let forward = right.cross(flow_dir).normalize();

        for _ in 0..count {
            let offset = match self.shape {
                FaucetShape::Circle { radius } => {
                    let r = radius * self.next_random().sqrt();
                    let theta = self.next_random() * 2.0 * PI;
                    right * (r * theta.cos()) + forward * (r * theta.sin())
                }
                FaucetShape::Rectangle { width, depth } => {
                    let rx = (self.next_random() - 0.5) * width;
                    let rz = (self.next_random() - 0.5) * depth;
                    right * rx + forward * rz
                }
                FaucetShape::Point => Vec3::ZERO,
            };

            let spawn_pos = self.position + offset;

            // Compute velocity with optional spray spread cone
            let vel = if self.spray_spread > 1e-4 {
                let spread_angle = self.spray_spread * self.next_random();
                let azim = self.next_random() * 2.0 * PI;
                let spread_dir = (flow_dir * spread_angle.cos()
                    + right * (spread_angle.sin() * azim.cos())
                    + forward * (spread_angle.sin() * azim.sin()))
                .normalize();
                spread_dir * speed
            } else {
                base_vel
            };

            particles.push(Particle::new(spawn_pos, vel, 1.0, *next_id));
            *next_id += 1;
        }

        particles
    }
}

/// Cube face where a drain can be installed
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeFace {
    Floor,   // -Y
    Ceiling, // +Y
    Left,    // -X
    Right,   // +X
    Back,    // -Z
    Front,   // +Z
    Custom,
}

impl CubeFace {
    pub const ALL: &'static [CubeFace] = &[
        CubeFace::Floor,
        CubeFace::Left,
        CubeFace::Right,
        CubeFace::Back,
        CubeFace::Front,
        CubeFace::Ceiling,
        CubeFace::Custom,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            CubeFace::Floor => "Floor (-Y)",
            CubeFace::Left => "Left Wall (-X)",
            CubeFace::Right => "Right Wall (+X)",
            CubeFace::Back => "Back Wall (-Z)",
            CubeFace::Front => "Front Wall (+Z)",
            CubeFace::Ceiling => "Ceiling (+Y)",
            CubeFace::Custom => "Custom 3D Position",
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            CubeFace::Floor => "floor",
            CubeFace::Left => "left",
            CubeFace::Right => "right",
            CubeFace::Back => "back",
            CubeFace::Front => "front",
            CubeFace::Ceiling => "ceiling",
            CubeFace::Custom => "custom",
        }
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "floor" | "bottom" => Some(CubeFace::Floor),
            "ceiling" | "top" => Some(CubeFace::Ceiling),
            "left" => Some(CubeFace::Left),
            "right" => Some(CubeFace::Right),
            "back" => Some(CubeFace::Back),
            "front" => Some(CubeFace::Front),
            "custom" => Some(CubeFace::Custom),
            _ => None,
        }
    }
}

/// Fluid drain / sink that eliminates particles passing through its aperture
#[derive(Clone, Debug)]
pub struct Drain {
    pub id: u32,
    pub name: String,
    pub face: CubeFace,
    pub position: Vec3,
    pub radius: f32,
    pub enabled: bool,
}

impl Drain {
    pub fn new(name: impl Into<String>, face: CubeFace, position: Vec3, radius: f32) -> Self {
        Self {
            id: 0,
            name: name.into(),
            face,
            position,
            radius,
            enabled: true,
        }
    }

    /// Check if particle at pos should be drained
    pub fn should_drain(&self, pos: Vec3, particle_radius: f32, bounds: &Aabb) -> bool {
        if !self.enabled {
            return false;
        }

        let r2 = self.radius * self.radius;
        let margin = particle_radius * 1.8;

        match self.face {
            CubeFace::Floor => {
                if pos.y <= bounds.min.y + margin {
                    let dx = pos.x - self.position.x;
                    let dz = pos.z - self.position.z;
                    if dx * dx + dz * dz <= r2 {
                        return true;
                    }
                }
            }
            CubeFace::Ceiling => {
                if pos.y >= bounds.max.y - margin {
                    let dx = pos.x - self.position.x;
                    let dz = pos.z - self.position.z;
                    if dx * dx + dz * dz <= r2 {
                        return true;
                    }
                }
            }
            CubeFace::Left => {
                if pos.x <= bounds.min.x + margin {
                    let dy = pos.y - self.position.y;
                    let dz = pos.z - self.position.z;
                    if dy * dy + dz * dz <= r2 {
                        return true;
                    }
                }
            }
            CubeFace::Right => {
                if pos.x >= bounds.max.x - margin {
                    let dy = pos.y - self.position.y;
                    let dz = pos.z - self.position.z;
                    if dy * dy + dz * dz <= r2 {
                        return true;
                    }
                }
            }
            CubeFace::Back => {
                if pos.z <= bounds.min.z + margin {
                    let dx = pos.x - self.position.x;
                    let dy = pos.y - self.position.y;
                    if dx * dx + dy * dy <= r2 {
                        return true;
                    }
                }
            }
            CubeFace::Front => {
                if pos.z >= bounds.max.z - margin {
                    let dx = pos.x - self.position.x;
                    let dy = pos.y - self.position.y;
                    if dx * dx + dy * dy <= r2 {
                        return true;
                    }
                }
            }
            CubeFace::Custom => {
                let dist_sq = (pos - self.position).length_squared();
                if dist_sq <= r2 {
                    return true;
                }
            }
        }

        false
    }
}
