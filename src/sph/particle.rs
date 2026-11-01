use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleType {
    Fluid,
    Boundary,
}

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub density: f32,
    pub pressure: f32,
    pub mass: f32,
    pub surface_normal: Vec3,
    pub surface_laplacian: f32,
    pub id: u32,
    pub particle_type: ParticleType,
}

impl Particle {
    pub fn new(position: Vec3, velocity: Vec3, mass: f32, id: u32) -> Self {
        Self {
            position,
            velocity,
            acceleration: Vec3::ZERO,
            density: 1000.0,
            pressure: 0.0,
            mass,
            surface_normal: Vec3::ZERO,
            surface_laplacian: 0.0,
            id,
            particle_type: ParticleType::Fluid,
        }
    }

    pub fn new_boundary(position: Vec3, mass: f32, id: u32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            density: 1000.0,
            pressure: 0.0,
            mass,
            surface_normal: Vec3::ZERO,
            surface_laplacian: 0.0,
            id,
            particle_type: ParticleType::Boundary,
        }
    }

    #[inline(always)]
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    #[inline(always)]
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.mass * self.velocity.length_squared()
    }
}
